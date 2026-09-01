/**
 * Client-side inference for the exported scikit-learn MLPRegressor.
 * Network math intentionally mirrors the existing production model.
 */
(function () {
  const DATA_BASE = "assets/data/";
  const FALLBACK_VALIDATION = { mae: 8480, rmse: 12023, r2: 0.8967, folds: 5 };
  const linScale = (v, min, max) => max === min ? 0 : (v - min) / (max - min);
  const linUnscale = (v, min, max) => v * (max - min) + min;
  function relu(v) { for (let i=0;i<v.length;i++) if(v[i]<0) v[i]=0; return v; }
  function dense(x,W,b) { const out=new Array(b.length).fill(0); for(let j=0;j<b.length;j++){let s=b[j];for(let i=0;i<x.length;i++)s+=x[i]*W[i][j];out[j]=s;} return out; }

  class CarModel {
    constructor(){ this.ready=this._load(); }
    async _load(){
      const [mr,dr]=await Promise.all([fetch(DATA_BASE+"model.json"),fetch(DATA_BASE+"dataset.json")]);
      if(!mr.ok||!dr.ok) throw new Error("Model assets could not be loaded.");
      this.model=await mr.json(); this.dataset=await dr.json();
      this.options=this.dataset.options; this.records=this.dataset.records; this.stats=this.dataset.stats;
      this.modelToManufacturer=this.dataset.model_to_manufacturer;
      this.modelToSubmodels=this.dataset.model_to_submodels;
      this.manufacturerToModels=this.dataset.manufacturer_to_models;
      // Prefer metadata exported alongside the model so running export_model.py
      // cannot silently revert the website to stale hard-coded metrics.
      this.validation=this.model.validation || this.dataset.validation || FALLBACK_VALIDATION;
      this.meta={hiddenLayers:this.model.architecture.hidden_layer_sizes,activation:this.model.architecture.activation,nFeatures:this.model.feature_order.length,nRows:this.stats.n_rows,validation:this.validation,groups:this.model.groups};
      this.datasetStats=this._stats(); return true;
    }
    predict(inputs){
      const m=this.model, ns={};
      m.numeric_cols.forEach((c,i)=>ns[c]=linScale(Number(inputs[c]),m.numeric_scaler.data_min[i],m.numeric_scaler.data_max[i]));
      const x=new Array(m.feature_order.length).fill(0);
      m.feature_order.forEach((col,i)=>{if(m.numeric_cols.includes(col)){x[i]=ns[col];return;} for(const g of Object.keys(m.groups)){const p=g+"_";if(col.startsWith(p)){if(inputs[g]===col.slice(p.length))x[i]=1;return;}}});
      let a=x; for(let l=0;l<m.weights.length;l++){a=dense(a,m.weights[l],m.biases[l]);if(l<m.weights.length-1)a=relu(a);}
      return {scaledPrice:a[0],price:linUnscale(a[0],m.price_scaler.data_min[0],m.price_scaler.data_max[0])};
    }
    comparableRecords(inputs,limit=6){
      const exact=this.records.filter(r=>r.manufacturer===inputs.manufacturer&&r.model===inputs.model);
      const pool=exact.length?exact:this.records.filter(r=>r.manufacturer===inputs.manufacturer);
      const source=pool.length?pool:this.records;
      const keys=["year","mileage","hand","engine_liters","horsepower"], ranges={};
      keys.forEach(k=>{const v=this.records.map(r=>Number(r[k]));ranges[k]=Math.max(...v)-Math.min(...v)||1;});
      return source.map(r=>{let d=0;keys.forEach(k=>d+=Math.abs(Number(r[k])-Number(inputs[k]))/ranges[k]);d/=keys.length;if(r.fuel!==inputs.fuel)d+=.08;if(r.transmission!==inputs.transmission)d+=.05;if(r.drive_type!==inputs.drive_type)d+=.05;if(r.submodel!==inputs.submodel)d+=.04;return {...r,_distance:d};}).sort((a,b)=>a._distance-b._distance).slice(0,limit);
    }
    marketPosition(inputs,price){
      const same=this.records.filter(r=>r.manufacturer===inputs.manufacturer&&r.model===inputs.model), pool=same.length>=8?same:this.records;
      return {percentile:Math.round(pool.filter(r=>r.price<price).length/pool.length*100),count:pool.length,scope:same.length>=8?`${inputs.manufacturer} ${inputs.model}`:"the full dataset",comparableCount:same.length};
    }
    coverage(inputs){
      const same=this.records.filter(r=>r.manufacturer===inputs.manufacturer&&r.model===inputs.model), nearest=this.comparableRecords(inputs,Math.min(12,this.records.length));
      const d=nearest.length?nearest.reduce((s,r)=>s+r._distance,0)/nearest.length:1;
      let level="Limited"; if(same.length>=20&&d<.16)level="High"; else if(same.length>=8&&d<.28)level="Medium";
      return {level,count:same.length,avgDistance:d};
    }
    _stats(){
      const r=this.records,p=r.map(x=>x.price).sort((a,b)=>a-b),med=p.length%2?p[(p.length-1)/2]:(p[p.length/2-1]+p[p.length/2])/2;
      const uniq=k=>new Set(r.map(x=>x[k])).size, avg=k=>r.reduce((s,x)=>s+Number(x[k]),0)/r.length;
      const count=k=>r.reduce((m,x)=>(m[x[k]]=(m[x[k]]||0)+1,m),{}), top=k=>Object.entries(count(k)).sort((a,b)=>b[1]-a[1]).slice(0,6);
      return {vehicles:r.length,manufacturers:uniq("manufacturer"),models:uniq("model"),averagePrice:avg("price"),medianPrice:med,averageMileage:avg("mileage"),priceMin:Math.min(...p),priceMax:Math.max(...p),topManufacturers:top("manufacturer"),topYears:top("year")};
    }
  }
  window.CarModel=new CarModel();
})();
