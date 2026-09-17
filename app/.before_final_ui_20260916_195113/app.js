(() => {
  "use strict";

  const DATA = window.PROCESSGAT || {classes:[],f1:{},support:{}};
  const GITHUB = "https://github.com/KiranJadhav09/ProcessGAT-XL";
  let trace = [];

  const $ = (s, root=document) => root.querySelector(s);
  const $$ = (s, root=document) => [...root.querySelectorAll(s)];

  function escapeHtml(v) {
    return String(v).replace(/[&<>"']/g, c => ({
      "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"
    }[c]));
  }

  function route(id, push=true) {
    const target = document.getElementById(id);
    if (!target) return;
    $$(".route").forEach(x => x.classList.remove("active"));
    target.classList.add("active");
    $$("[data-route]").forEach(x => x.classList.toggle("active", x.dataset.route === id));
    $("#mobileNav")?.classList.remove("open");
    $("#menuButton")?.setAttribute("aria-expanded","false");
    window.scrollTo({top:0,behavior:"smooth"});
    if (push) history.pushState({route:id},"","#"+id);
    observeReveals();
  }

  function renderTrace() {
    const el = $("#trace");
    if (!el) return;
    if (!trace.length) {
      el.innerHTML = '<span class="trace-empty">Your observed case will appear here.</span>';
      return;
    }
    el.innerHTML = trace.map((x,i) =>
      `<div class="chip"><span>${String(i+1).padStart(2,"0")}</span>${escapeHtml(x)}<button type="button" data-remove="${i}" aria-label="Remove activity">×</button></div>`
    ).join("");
  }

  function addActivity(activity) {
    if (!activity) return;
    trace.push(activity);
    renderTrace();
  }

  function removeActivity(i) {
    const chip = $$(".chip")[i];
    if (chip) chip.classList.add("removing");
    setTimeout(() => {
      trace.splice(i,1);
      renderTrace();
    },150);
  }

  function setPlaceholder() {
    const out=$("#predictionList");
    if (!out) return;
    out.innerHTML = '<div class="empty-output"><div class="empty-icon">→</div><span>Prediction results will appear here.</span></div>';
  }

  function normalisePredictions(data) {
    let arr = data?.predictions ?? data?.results ?? data?.prediction ?? [];
    if (!Array.isArray(arr) && arr && typeof arr === "object") {
      arr = Object.entries(arr).map(([activity,probability]) => ({activity,probability}));
    }
    return arr.map((x, i) => ({
      activity: x.activity ?? x.label ?? x.class ?? x.name ?? x[0] ?? `Class ${i+1}`,
      probability: Number(x.probability ?? x.confidence ?? x.score ?? x[1] ?? 0)
    })).filter(x => Number.isFinite(x.probability)).sort((a,b)=>b.probability-a.probability).slice(0,3);
  }

  async function predict() {
    const status=$("#predictionStatus"), out=$("#predictionList"), btn=$("#predictButton");
    if (!trace.length) {
      status.textContent="Add at least one activity to the trace.";
      return;
    }

    btn.disabled=true;
    btn.classList.add("loading");
    btn.innerHTML='<span class="spinner"></span> Predicting…';
    status.textContent="Processing the case history…";
    out.innerHTML="";

    const started=performance.now();

    try {
      const response=await fetch("/predict",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({trace})
      });
      const data=await response.json();
      if (!response.ok) throw new Error(data.detail || "The model could not process this trace.");

      const predictions=normalisePredictions(data);
      if (!predictions.length) throw new Error("The model returned no prediction.");

      const elapsed=Number(data.inference_ms ?? (performance.now()-started));
      status.textContent="Prediction complete.";
      $("#inferenceTime").textContent=`${elapsed.toFixed(1)} ms`;

      out.innerHTML=predictions.map((x,i)=>{
        const pct=Math.max(0,Math.min(100,x.probability*100));
        return `<div class="prediction-row" style="--delay:${i*90}ms">
          <span class="rank">${String(i+1).padStart(2,"0")}</span>
          <div><b>${escapeHtml(x.activity)}</b><div class="bar"><i data-width="${pct}"></i></div></div>
          <strong>${pct.toFixed(2)}%</strong>
        </div>`;
      }).join("");

      requestAnimationFrame(() => setTimeout(() => {
        $$("#predictionList .bar i").forEach(x => x.style.width=x.dataset.width+"%");
      },50));
    } catch (error) {
      status.textContent=error.message || "Unable to reach the model.";
      $("#inferenceTime").textContent="—";
      out.innerHTML=`<div class="empty-output"><div class="empty-icon">!</div><span>${escapeHtml(status.textContent)}</span></div>`;
    } finally {
      btn.disabled=false;
      btn.classList.remove("loading");
      btn.innerHTML='Predict next activity <span>→</span>';
    }
  }

  function loadClasses() {
    const list=$("#activityButtons");
    if (!list) return;
    list.innerHTML=DATA.classes.map((x,i)=>
      `<button type="button" class="activity-btn" data-activity="${escapeHtml(x)}"><span>${String(i+1).padStart(2,"0")}</span>${escapeHtml(x)}</button>`
    ).join("");
  }

  function loadClassTable() {
    const tbody=$("#classTable tbody");
    if (!tbody) return;
    tbody.innerHTML=Object.keys(DATA.f1).map(name=>
      `<tr><td>${escapeHtml(name)}</td><td>${(Number(DATA.f1[name])*100).toFixed(2)}%</td><td>${Number(DATA.support[name]||0).toLocaleString()}</td></tr>`
    ).join("");
  }

  function countUp(el) {
    if (el.dataset.counted) return;
    el.dataset.counted="1";
    const target=Number(el.dataset.count), start=performance.now(), duration=1000;
    const tick=now=>{
      const p=Math.min(1,(now-start)/duration);
      const eased=1-Math.pow(1-p,3);
      el.textContent=Math.floor(target*eased).toLocaleString();
      if (p<1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  let observer;
  function observeReveals() {
    if (observer) observer.disconnect();
    observer=new IntersectionObserver(entries=>{
      entries.forEach(entry=>{
        if (!entry.isIntersecting) return;
        entry.target.classList.add("in");
        if (entry.target.matches("[data-count]")) countUp(entry.target);
        observer.unobserve(entry.target);
      });
    },{threshold:.15});
    $$(".reveal,.metric strong,.three-up article,.result-metrics,.evaluation-grid article").forEach(x=>observer.observe(x));
  }

  function progress() {
    const scrollTop=window.scrollY;
    const height=document.documentElement.scrollHeight-window.innerHeight;
    $("#appProgress").style.width=(height>0?(scrollTop/height)*100:0)+"%";
  }

  document.addEventListener("click", e=>{
    const routeButton=e.target.closest("[data-route]");
    if (routeButton) {
      e.preventDefault();
      route(routeButton.dataset.route);
      return;
    }
    const activity=e.target.closest("[data-activity]");
    if (activity) {
      addActivity(activity.dataset.activity);
      return;
    }
    const remove=e.target.closest("[data-remove]");
    if (remove) {
      removeActivity(Number(remove.dataset.remove));
      return;
    }
    const example=e.target.closest("[data-example]");
    if (example) {
      const examples={
        1:["W_Completeren aanvraag"],
        2:["W_Completeren aanvraag","W_Nabellen offertes"],
        3:["W_Beoordelen fraude","W_Nabellen incomplete dossiers"],
        4:["W_Valideren aanvraag"]
      };
      trace=[...(examples[example.dataset.example]||[])];
      renderTrace();
      route("prediction");
    }
  });

  document.addEventListener("DOMContentLoaded",()=>{
    loadClasses();
    loadClassTable();
    renderTrace();

    $("#clearTrace")?.addEventListener("click",()=>{
      trace=[];
      renderTrace();
      $("#predictionStatus").textContent="Awaiting a trace.";
      $("#inferenceTime").textContent="—";
      setPlaceholder();
    });

    $("#predictButton")?.addEventListener("click",predict);

    $("#menuButton")?.addEventListener("click",()=>{
      const nav=$("#mobileNav");
      const open=!nav.classList.contains("open");
      nav.classList.toggle("open",open);
      $("#menuButton").setAttribute("aria-expanded",String(open));
    });

    window.addEventListener("popstate",()=>{
      const id=location.hash.replace("#","");
      route(document.getElementById(id)?id:"home",false);
    });

    document.addEventListener("mousedown",e=>{
      const b=e.target.closest("button,.button");
      if (b) b.classList.add("pressed");
    });
    document.addEventListener("mouseup",e=>{
      const b=e.target.closest("button,.button");
      if (b) b.classList.remove("pressed");
    });

    const hash=location.hash.replace("#","");
    if (hash && document.getElementById(hash)) route(hash,false);
    observeReveals();
    progress();
    window.addEventListener("scroll",progress,{passive:true});
  });
})();
