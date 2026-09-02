/**
 * Client-side inference for the exported scikit-learn MLPRegressor.
 * Network math intentionally mirrors the existing production model.
 */
(function () {
  const DATA_BASES = ["assets/data/", "../data/"];
  const FALLBACK_VALIDATION = { mae: 8480, rmse: 12023, r2: 0.8967, folds: 5 };
  
  // Linear scaling functions matching scikit-learn's MinMaxScaler
  const linScale = (v, min, max) => max === min ? 0 : (v - min) / (max - min);
  const linUnscale = (v, min, max) => v * (max - min) + min;
  
  // Rectified Linear Unit (ReLU) activation function
  function relu(v) { 
    for (let i = 0; i < v.length; i++) {
        if (v[i] < 0) {
            v[i] = 0; 
        }
    }
    return v; 
  }
  
  // Dense (fully connected) layer math implementation
  function dense(x, W, b) { 
    const out = new Array(b.length).fill(0); 
    for (let j = 0; j < b.length; j++) {
        let sum = b[j]; // Start with the bias
        for (let i = 0; i < x.length; i++) {
            sum += x[i] * W[i][j]; // Add weight * input
        }
        out[j] = sum;
    } 
    return out; 
  }

  class CarModel {
    constructor(){ 
        this.ready = this._load(); 
    }

    // Load model weights and dataset JSONs
    async _load(){
      let mr = null;
      let dr = null;
      let lastError = null;

      for (const base of DATA_BASES) {
        try {
          const modelRes = fetch(base + "model.json", { cache: "no-store" });
          const datasetRes = fetch(base + "dataset.json", { cache: "no-store" });
          const pair = await Promise.all([modelRes, datasetRes]);
          
          if (pair[0].ok && pair[1].ok) {
              mr = pair[0];
              dr = pair[1];
              break;
          }
          lastError = new Error(`Model assets returned HTTP ${pair[0].status}/${pair[1].status} from ${base}`);
        } catch(err) {
            lastError = err;
        }
      }

      if (!mr || !dr) {
          throw (lastError || new Error("Model assets could not be loaded."));
      }

      this.model = await mr.json(); 
      this.dataset = await dr.json();
      this.options = this.dataset.options; 
      this.records = this.dataset.records; 
      this.stats = this.dataset.stats;
      this.modelToManufacturer = this.dataset.model_to_manufacturer;
      this.modelToSubmodels = this.dataset.model_to_submodels;
      this.manufacturerToModels = this.dataset.manufacturer_to_models;
      this.validation = this.model.validation || this.dataset.validation || FALLBACK_VALIDATION;
      
      this.meta = {
          hiddenLayers: this.model.architecture.hidden_layer_sizes,
          activation: this.model.architecture.activation,
          nFeatures: this.model.feature_order.length,
          nRows: this.stats.n_rows,
          validation: this.validation,
          groups: this.model.groups
      };
      
      this.datasetStats = this._stats(); 
      return true;
    }

    // Run inference (prediction) using the loaded weights
    predict(inputs){
      const m = this.model;
      const ns = {};
      
      // Scale numeric inputs
      m.numeric_cols.forEach((c, i) => {
          ns[c] = linScale(Number(inputs[c]), m.numeric_scaler.data_min[i], m.numeric_scaler.data_max[i]);
      });
      
      // Build the input vector mapping categorical/one-hot variables
      const x = new Array(m.feature_order.length).fill(0);
      m.feature_order.forEach((col, i) => {
          if (m.numeric_cols.includes(col)) {
              x[i] = ns[col];
              return;
          } 
          for (const g of Object.keys(m.groups)) {
              const prefix = g + "_";
              if (col.startsWith(prefix)) {
                  if (inputs[g] === col.slice(prefix.length)) {
                      x[i] = 1;
                  }
                  return;
              }
          }
      });
      
      // Forward pass through the network layers
      let a = x; 
      for (let l = 0; l < m.weights.length; l++) {
          a = dense(a, m.weights[l], m.biases[l]);
          if (l < m.weights.length - 1) {
              a = relu(a);
          }
      }
      
      // Unscale output back to actual currency amount
      return {
          scaledPrice: a[0],
          price: linUnscale(a[0], m.price_scaler.data_min[0], m.price_scaler.data_max[0])
      };
    }

    // Find similar cars based on distance metrics
    comparableRecords(inputs, limit = 6){
      const exact = this.records.filter(r => r.manufacturer === inputs.manufacturer && r.model === inputs.model);
      const pool = exact.length ? exact : this.records.filter(r => r.manufacturer === inputs.manufacturer);
      const source = pool.length ? pool : this.records;
      
      const keys = ["year", "mileage", "hand", "engine_liters", "horsepower"];
      const ranges = {};
      
      keys.forEach(k => {
          const v = this.records.map(r => Number(r[k]));
          ranges[k] = Math.max(...v) - Math.min(...v) || 1;
      });
      
      // Calculate a distance score for each comparable car
      return source.map(r => {
          let d = 0;
          keys.forEach(k => {
              d += Math.abs(Number(r[k]) - Number(inputs[k])) / ranges[k];
          });
          d /= keys.length;
          
          if (r.fuel !== inputs.fuel) d += 0.08;
          if (r.transmission !== inputs.transmission) d += 0.05;
          if (r.drive_type !== inputs.drive_type) d += 0.05;
          if (r.submodel !== inputs.submodel) d += 0.04;
          
          return { ...r, _distance: d };
      }).sort((a, b) => a._distance - b._distance).slice(0, limit);
    }

    // Determine how the car sits compared to market data
    marketPosition(inputs, price){
      const same = this.records.filter(r => r.manufacturer === inputs.manufacturer && r.model === inputs.model);
      const pool = same.length >= 8 ? same : this.records;
      const lowerPricedCount = pool.filter(r => r.price < price).length;
      
      return {
          percentile: Math.round((lowerPricedCount / pool.length) * 100),
          count: pool.length,
          scope: same.length >= 8 ? `${inputs.manufacturer} ${inputs.model}` : "the full dataset",
          comparableCount: same.length
      };
    }

    // Determine reliability based on dataset representation
    coverage(inputs){
      const same = this.records.filter(r => r.manufacturer === inputs.manufacturer && r.model === inputs.model);
      const nearest = this.comparableRecords(inputs, Math.min(12, this.records.length));
      
      let totalDistance = 0;
      nearest.forEach(r => totalDistance += r._distance);
      const d = nearest.length ? (totalDistance / nearest.length) : 1;
      
      let level = "Limited"; 
      if (same.length >= 20 && d < 0.16) {
          level = "High";
      } else if (same.length >= 8 && d < 0.28) {
          level = "Medium";
      }
      
      return { level, count: same.length, avgDistance: d };
    }

    // Generate summary stats for the dashboard view
    _stats(){
      const r = this.records;
      const prices = r.map(x => x.price).sort((a, b) => a - b);
      
      let med = 0;
      const mid = Math.floor(prices.length / 2);
      if (prices.length % 2 !== 0) {
          med = prices[mid];
      } else {
          med = (prices[mid - 1] + prices[mid]) / 2;
      }
      
      const uniq = k => new Set(r.map(x => x[k])).size;
      const avg = k => r.reduce((sum, x) => sum + Number(x[k]), 0) / r.length;
      
      const count = k => r.reduce((map, x) => {
          map[x[k]] = (map[x[k]] || 0) + 1;
          return map;
      }, {});
      
      const top = k => Object.entries(count(k))
          .sort((a, b) => b[1] - a[1])
          .slice(0, 6);
          
      return {
          vehicles: r.length,
          manufacturers: uniq("manufacturer"),
          models: uniq("model"),
          averagePrice: avg("price"),
          medianPrice: med,
          averageMileage: avg("mileage"),
          priceMin: Math.min(...prices),
          priceMax: Math.max(...prices),
          topManufacturers: top("manufacturer"),
          topYears: top("year")
      };
    }
  }
  
  window.CarModel = new CarModel();
})();