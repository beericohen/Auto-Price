/**
 * model.js
 * --------
 * Runs the trained MLPRegressor entirely in the browser, mirroring
 * scikit-learn's MinMaxScaler + MLPRegressor.predict() math exactly
 * (verified against the original Python pipeline row-for-row).
 *
 * Exposes `window.CarModel` with:
 *   .ready              -> Promise, resolves once model.json + dataset.json load
 *   .predict(inputs)    -> { price, scaledPrice }
 *   .options            -> category lists + cascading maps (from dataset.json)
 *   .records            -> decoded training rows (for the scatter chart)
 *   .stats              -> dataset-level min/max stats
 *   .meta               -> architecture + fit metrics for the About section
 */
(function () {
  const DATA_BASE = "assets/data/";

  function linScale(value, dataMin, dataMax) {
    if (dataMax === dataMin) return 0;
    return (value - dataMin) / (dataMax - dataMin);
  }

  function linUnscale(value, dataMin, dataMax) {
    return value * (dataMax - dataMin) + dataMin;
  }

  function relu(vec) {
    for (let i = 0; i < vec.length; i++) if (vec[i] < 0) vec[i] = 0;
    return vec;
  }

  // out = x (1xN) @ W (NxM) + b (M)
  function denseForward(x, W, b) {
    const out = new Array(b.length).fill(0);
    for (let j = 0; j < b.length; j++) {
      let sum = b[j];
      for (let i = 0; i < x.length; i++) {
        sum += x[i] * W[i][j];
      }
      out[j] = sum;
    }
    return out;
  }

  class CarModel {
    constructor() {
      this.ready = this._load();
    }

    async _load() {
      const [modelRes, dataRes] = await Promise.all([
        fetch(DATA_BASE + "model.json"),
        fetch(DATA_BASE + "dataset.json"),
      ]);
      if (!modelRes.ok || !dataRes.ok) {
        throw new Error("Could not load model data. Serve this site over http(s), not file://.");
      }
      this.model = await modelRes.json();
      this.dataset = await dataRes.json();

      this.options = this.dataset.options;
      this.records = this.dataset.records;
      this.stats = this.dataset.stats;
      this.modelToManufacturer = this.dataset.model_to_manufacturer;
      this.modelToSubmodels = this.dataset.model_to_submodels;
      this.manufacturerToModels = this.dataset.manufacturer_to_models;

      const arch = this.model.architecture;
      // in-sample fit metrics, computed once at export time in Python
      this.meta = {
        hiddenLayers: arch.hidden_layer_sizes,
        activation: arch.activation,
        nFeatures: this.model.feature_order.length,
        nRows: this.stats.n_rows,
        groups: this.model.groups,
      };

      return true;
    }

    /**
     * inputs: {
     *   manufacturer, model, submodel, fuel, transmission, drive_type,  // strings
     *   year, hand, engine_liters, horsepower, mileage                  // numbers
     * }
     */
    predict(inputs) {
      const m = this.model;
      const featureOrder = m.feature_order;
      const numericCols = m.numeric_cols;
      const scaler = m.numeric_scaler;
      const priceScaler = m.price_scaler;

      // Build a lookup of numeric scaled values
      const numericScaled = {};
      numericCols.forEach((col, idx) => {
        const raw = Number(inputs[col]);
        numericScaled[col] = linScale(raw, scaler.data_min[idx], scaler.data_max[idx]);
      });

      // Build the one-hot feature vector in the exact training column order
      const x = new Array(featureOrder.length).fill(0);
      featureOrder.forEach((col, idx) => {
        if (numericCols.includes(col)) {
          x[idx] = numericScaled[col];
          return;
        }
        for (const groupKey of Object.keys(m.groups)) {
          const prefix = groupKey + "_";
          if (col.startsWith(prefix)) {
            const value = col.slice(prefix.length);
            if (inputs[groupKey] === value) x[idx] = 1;
            return;
          }
        }
      });

      // Forward pass: Dense -> ReLU -> Dense -> ReLU -> Dense (identity out)
      let a = x;
      for (let layer = 0; layer < m.weights.length; layer++) {
        a = denseForward(a, m.weights[layer], m.biases[layer]);
        if (layer < m.weights.length - 1) a = relu(a);
      }
      const scaledPrice = a[0];
      const price = linUnscale(scaledPrice, priceScaler.data_min[0], priceScaler.data_max[0]);

      return { price, scaledPrice };
    }
  }

  window.CarModel = new CarModel();
})();
