const CLASSES = __CLASSES__;
const F1 = __F1__;
const SUPPORT = __SUPPORT__;
let trace = [];

function go(page) {
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  document.querySelectorAll("[data-page]").forEach(b => b.classList.toggle("active", b.dataset.page === page));
  const target = document.getElementById(page);
  if (target) target.classList.add("active");
  document.getElementById("mobileNav")?.classList.remove("open");
  window.scrollTo({top:0, behavior:"smooth"});
}

function renderTrace() {
  const el = document.getElementById("trace");
  if (!el) return;
  el.innerHTML = trace.length
    ? trace.map((x,i) => `<div class="activity">${i+1} · ${x} <button class="remove" onclick="removeActivity(${i})">×</button></div>`).join("")
    : `<span class="muted">No activities added.</span>`;
}

function removeActivity(i) { trace.splice(i,1); renderTrace(); }

function clearTrace() {
  trace = [];
  renderTrace();
  document.getElementById("predictionStatus").textContent = "Awaiting a process trace.";
  document.getElementById("prediction").innerHTML = "";
}

function sample(n) {
  const examples = {
    1:["W_Completeren aanvraag"],
    2:["W_Completeren aanvraag","W_Nabellen offertes"],
    3:["W_Beoordelen fraude","W_Nabellen incomplete dossiers"],
    4:["W_Valideren aanvraag"]
  };
  trace = [...examples[n]];
  renderTrace();
  go("predict");
}

async function runPrediction() {
  const status = document.getElementById("predictionStatus");
  const output = document.getElementById("prediction");
  if (!trace.length) {
    status.textContent = "Add at least one activity.";
    return;
  }
  status.textContent = "Running inference…";
  output.innerHTML = "";
  try {
    const r = await fetch("/predict", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({trace})
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "Prediction failed");
    let a = d.predictions || d.results || [];
    if (!Array.isArray(a) && typeof a === "object") {
      a = Object.entries(a).map(([activity, probability]) => ({activity, probability}));
    }
    a = a.map(x => ({
      activity:x.activity || x.label || x.class || x[0],
      probability:Number(x.probability ?? x.confidence ?? x.score ?? x[1])
    })).sort((x,y)=>y.probability-x.probability).slice(0,3);
    status.textContent = `Inference completed · ${Number(d.inference_ms || 0).toFixed(1)} ms`;
    output.innerHTML = a.map((x,i) =>
      `<div class="rank"><b>#${i+1}</b><div>${x.activity}<div class="bar"><i style="width:${Math.min(100,x.probability*100)}%"></i></div></div><b>${(x.probability*100).toFixed(2)}%</b></div>`
    ).join("");
  } catch(e) {
    status.textContent = e.message;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-page]").forEach(b => b.addEventListener("click", () => go(b.dataset.page)));
  document.getElementById("mobileMenu")?.addEventListener("click", () => document.getElementById("mobileNav").classList.toggle("open"));
  const select = document.getElementById("activity");
  CLASSES.forEach(x => { const o=document.createElement("option"); o.value=x; o.textContent=x; select?.appendChild(o); });
  document.getElementById("addActivity")?.addEventListener("click", () => {
    if (select.value) { trace.push(select.value); select.value=""; renderTrace(); }
  });
  document.getElementById("clearTrace")?.addEventListener("click", clearTrace);
  document.getElementById("predictButton")?.addEventListener("click", runPrediction);
  document.querySelectorAll("[data-example]").forEach(b => b.addEventListener("click", () => sample(Number(b.dataset.example))));
  const tbody=document.querySelector("#classTable tbody");
  if(tbody) Object.keys(F1).forEach(k => {
    const tr=document.createElement("tr");
    tr.innerHTML=`<td>${k}</td><td>${(F1[k]*100).toFixed(2)}%</td><td>${SUPPORT[k].toLocaleString()}</td>`;
    tbody.appendChild(tr);
  });
  renderTrace();
});
