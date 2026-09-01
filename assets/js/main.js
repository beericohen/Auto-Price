import { initCar3D } from "./car3d.js";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function formatShekelWhole(n) {
  return Math.round(n).toLocaleString("en-US");
}

function fillSelect(select, values, { placeholder } = {}) {
  select.innerHTML = "";
  if (placeholder) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = placeholder;
    opt.disabled = true;
    opt.selected = true;
    select.appendChild(opt);
  }
  values.forEach((v) => {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    select.appendChild(opt);
  });
}

function setupSliderFill(input) {
  const update = () => {
    const min = Number(input.min);
    const max = Number(input.max);
    const pct = ((Number(input.value) - min) / (max - min)) * 100;
    input.style.setProperty("--fill", pct + "%");
  };
  input.addEventListener("input", update);
  update();
}

/* ---------------------------------------------------------------------- */
/* Odometer                                                                */
/* ---------------------------------------------------------------------- */

class Odometer {
  constructor(container, digitCount) {
    this.container = container;
    this.digitCount = digitCount;
    this.digits = [];
    this.container.innerHTML = "";
    for (let i = 0; i < digitCount; i++) {
      const digit = document.createElement("span");
      digit.className = "odometer-digit";
      const reel = document.createElement("span");
      reel.className = "reel";
      for (let n = 0; n <= 9; n++) {
        const s = document.createElement("span");
        s.textContent = n;
        reel.appendChild(s);
      }
      digit.appendChild(reel);
      this.container.appendChild(digit);
      this.digits.push({ el: digit, reel });
    }
  }

  set(value) {
    const str = String(Math.max(0, Math.round(value))).padStart(this.digitCount, "0").slice(-this.digitCount);
    str.split("").forEach((ch, i) => {
      const n = Number(ch);
      const { reel } = this.digits[i];
      reel.style.transform = `translateY(${-n * (100 / 10)}%)`;
    });
  }
}

/* ---------------------------------------------------------------------- */
/* Boot                                                                    */
/* ---------------------------------------------------------------------- */

async function boot() {
  const statusEl = $("#load-status");
  try {
    await window.CarModel.ready;
  } catch (err) {
    if (statusEl) {
      statusEl.textContent =
        "Couldn't load the model data. If you opened this file directly, serve it with a local server or GitHub Pages instead — browsers block fetch() on file://.";
      statusEl.style.display = "block";
    }
    console.error(err);
    return;
  }

  const CarModel = window.CarModel;
  const { options, manufacturerToModels, modelToSubmodels } = CarModel;

  // ---- 3D car ----
  const carCanvas = $("#car3d-canvas");
  let car3d = null;
  if (carCanvas) car3d = initCar3D(carCanvas);

  // ---- populate About section stats ----
  $("#stat-r2").textContent = "0.9";
  $("#stat-mae").textContent = "\u20AA" + formatShekelWhole(7973);
  $("#stat-rows").textContent = CarModel.stats.n_rows.toLocaleString("en-US");
  $("#stat-features").textContent = String(CarModel.meta.nFeatures);
  $("#stat-rows-2").textContent = CarModel.stats.n_rows.toLocaleString("en-US");
  const [h1, h2] = CarModel.meta.hiddenLayers;
  if ($("#stat-hidden-1")) $("#stat-hidden-1").textContent = String(h1);
  if ($("#stat-hidden-2")) $("#stat-hidden-2").textContent = String(h2);

  // ---- form elements ----
  const form = $("#valuation-form");
  const manufacturerSel = $("#f-manufacturer");
  const modelSel = $("#f-model");
  const submodelSel = $("#f-submodel");
  const fuelSel = $("#f-fuel");
  const transmissionSel = $("#f-transmission");
  const driveSel = $("#f-drive");

  const yearInput = $("#f-year");
  const handInput = $("#f-hand");
  const mileageInput = $("#f-mileage");
  const engineInput = $("#f-engine");
  const hpInput = $("#f-horsepower");

  fillSelect(manufacturerSel, options.manufacturer, { placeholder: "Choose manufacturer" });
  fillSelect(fuelSel, options.fuel);
  fillSelect(transmissionSel, options.transmission);
  fillSelect(driveSel, options.drive_type);
  modelSel.disabled = true;
  submodelSel.disabled = true;
  fillSelect(modelSel, [], { placeholder: "Choose manufacturer first" });
  fillSelect(submodelSel, [], { placeholder: "Choose model first" });

  [yearInput, handInput, mileageInput, engineInput, hpInput].forEach((input) => {
    setupSliderFill(input);
    const out = $(`[data-out-for="${input.id}"]`);
    const render = () => {
      if (!out) return;
      if (input === mileageInput) out.textContent = Number(input.value).toLocaleString("en-US") + " km";
      else if (input === engineInput) out.textContent = Number(input.value).toFixed(1) + " L";
      else if (input === hpInput) out.textContent = input.value + " hp";
      else if (input === handInput) out.textContent = input.value + (input.value === "1" ? " owner" : " owners");
      else out.textContent = input.value;
    };
    input.addEventListener("input", render);
    render();
  });

  manufacturerSel.addEventListener("change", () => {
    const manu = manufacturerSel.value;
    const models = manufacturerToModels[manu] || ["Other"];
    modelSel.disabled = false;
    fillSelect(modelSel, models, { placeholder: "Choose model" });
    submodelSel.disabled = true;
    fillSelect(submodelSel, [], { placeholder: "Choose model first" });
  });

  modelSel.addEventListener("change", () => {
    const modelName = modelSel.value;
    const submodels = modelToSubmodels[modelName] || options.submodel;
    submodelSel.disabled = false;
    fillSelect(submodelSel, submodels, { placeholder: "Choose submodel" });
  });

  fuelSel.addEventListener("change", () => {
    if (car3d) car3d.setFuelColor(fuelSel.value);
  });
  if (car3d) car3d.setFuelColor(fuelSel.value);

  // ---- odometer ----
  const odometer = new Odometer($("#odometer-digits"), 6);
  odometer.set(0);
  const resultContext = $("#result-context");
  const resultPlaceholder = $("#result-placeholder");
  const resultConfidence = $("#result-confidence");

  // ---- chart ----
  const chartSvg = $("#scatter-svg");
  const chart = new window.ScatterChart(chartSvg, CarModel.records, CarModel.stats);
  $$(".axis-toggle button").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".axis-toggle button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      chart.setAxis(btn.dataset.axis);
    });
  });

  function percentileContext(inputs, price) {
    const sameModel = CarModel.records.filter(
      (r) => r.manufacturer === inputs.manufacturer && r.model === inputs.model
    );
    const pool = sameModel.length >= 8 ? sameModel : CarModel.records;
    const cheaperCount = pool.filter((r) => r.price < price).length;
    const percentile = Math.round((cheaperCount / pool.length) * 100);
    const scope = sameModel.length >= 8 ? `similar ${inputs.manufacturer} ${inputs.model} listings` : "listings in the dataset";
    return `That's higher than ${percentile}% of ${scope}.`;
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();

    const inputs = {
      manufacturer: manufacturerSel.value,
      model: modelSel.value,
      submodel: submodelSel.value,
      fuel: fuelSel.value,
      transmission: transmissionSel.value,
      drive_type: driveSel.value,
      year: Number(yearInput.value),
      hand: Number(handInput.value),
      engine_liters: Number(engineInput.value),
      horsepower: Number(hpInput.value),
      mileage: Number(mileageInput.value),
    };

    if (!inputs.manufacturer || !inputs.model || !inputs.submodel) {
      resultPlaceholder.textContent = "Fill in manufacturer, model, and submodel to get an estimate.";
      resultPlaceholder.style.display = "block";
      return;
    }
    resultPlaceholder.style.display = "none";

    const { price } = CarModel.predict(inputs);
    const clamped = Math.max(0, price);
    odometer.set(clamped);
    resultContext.textContent = percentileContext(inputs, clamped);
    resultConfidence.textContent = `Typical error on the training data: about \u20AA${formatShekelWhole(8000)}. Treat this as a starting point for negotiation, not a final price.`;

    chart.setYou({ mileage: inputs.mileage, year: inputs.year, price: clamped });
    $("#market").scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

boot();
