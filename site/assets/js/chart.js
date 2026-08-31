/**
 * chart.js
 * --------
 * Lightweight SVG scatter chart (no dependency) plotting price against
 * either mileage or year, with the market as translucent teal dots and
 * the user's predicted car as a single amber pin.
 */
(function () {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const WIDTH = 860;
  const HEIGHT = 440;
  const PAD = { top: 20, right: 24, bottom: 44, left: 66 };

  function el(tag, attrs) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    return node;
  }

  function formatShekel(n) {
    return "\u20AA" + Math.round(n).toLocaleString("en-US");
  }

  class ScatterChart {
    constructor(svgEl, records, priceRange) {
      this.svg = svgEl;
      this.records = records;
      this.priceRange = priceRange;
      this.axis = "mileage"; // 'mileage' | 'year'
      this.you = null; // { mileage, year, price }
      this.svg.setAttribute("viewBox", `0 0 ${WIDTH} ${HEIGHT}`);
      this._computeScales();
      this.render();
    }

    setAxis(axis) {
      this.axis = axis;
      this._computeScales();
      this.render();
    }

    setYou(point) {
      this.you = point;
      this.render();
    }

    _computeScales() {
      const xs = this.records.map((r) => r[this.axis]);
      if (this.you) xs.push(this.you[this.axis]);
      const ys = this.records.map((r) => r.price);
      if (this.you) ys.push(this.you.price);

      const xMin = Math.min(...xs);
      const xMax = Math.max(...xs);
      const yMin = 0;
      const yMax = Math.max(...ys) * 1.05;

      const innerW = WIDTH - PAD.left - PAD.right;
      const innerH = HEIGHT - PAD.top - PAD.bottom;

      this.scaleX = (v) => PAD.left + ((v - xMin) / (xMax - xMin || 1)) * innerW;
      this.scaleY = (v) => PAD.top + innerH - ((v - yMin) / (yMax - yMin || 1)) * innerH;
      this.xMin = xMin;
      this.xMax = xMax;
      this.yMax = yMax;
    }

    _axisTicks(min, max, count) {
      const ticks = [];
      const step = (max - min) / (count - 1);
      for (let i = 0; i < count; i++) ticks.push(min + step * i);
      return ticks;
    }

    render() {
      while (this.svg.firstChild) this.svg.removeChild(this.svg.firstChild);

      const innerW = WIDTH - PAD.left - PAD.right;
      const innerH = HEIGHT - PAD.top - PAD.bottom;

      // gridlines + y ticks (price)
      const yTicks = this._axisTicks(0, this.yMax, 5);
      yTicks.forEach((t) => {
        const y = this.scaleY(t);
        this.svg.appendChild(
          el("line", {
            x1: PAD.left,
            x2: PAD.left + innerW,
            y1: y,
            y2: y,
            stroke: "#2b3140",
            "stroke-width": 1,
            "stroke-dasharray": "2 5",
            opacity: 0.6,
          })
        );
        const label = el("text", {
          x: PAD.left - 10,
          y: y + 4,
          "text-anchor": "end",
          fill: "#8b93a6",
          "font-size": 11,
          "font-family": "IBM Plex Sans, sans-serif",
        });
        label.textContent = formatShekel(t);
        this.svg.appendChild(label);
      });

      // x ticks
      const xTicks = this._axisTicks(this.xMin, this.xMax, 6);
      xTicks.forEach((t) => {
        const x = this.scaleX(t);
        const label = el("text", {
          x: x,
          y: HEIGHT - PAD.bottom + 22,
          "text-anchor": "middle",
          fill: "#8b93a6",
          "font-size": 11,
          "font-family": "IBM Plex Sans, sans-serif",
        });
        label.textContent =
          this.axis === "mileage" ? Math.round(t / 1000) + "k" : Math.round(t);
        this.svg.appendChild(label);
      });

      const axisLabel = el("text", {
        x: PAD.left + innerW / 2,
        y: HEIGHT - 6,
        "text-anchor": "middle",
        fill: "#5b6273",
        "font-size": 11,
        "letter-spacing": "0.06em",
        "font-family": "Rajdhani, sans-serif",
      });
      axisLabel.textContent = this.axis === "mileage" ? "MILEAGE (KM)" : "YEAR";
      this.svg.appendChild(axisLabel);

      // market dots
      const dotsGroup = el("g", {});
      this.records.forEach((r) => {
        const c = el("circle", {
          cx: this.scaleX(r[this.axis]),
          cy: this.scaleY(r.price),
          r: 2.6,
          fill: "#35d6b3",
          opacity: 0.35,
        });
        dotsGroup.appendChild(c);
      });
      this.svg.appendChild(dotsGroup);

      // your car
      if (this.you) {
        const cx = this.scaleX(this.you[this.axis]);
        const cy = this.scaleY(this.you.price);

        const halo = el("circle", {
          cx,
          cy,
          r: 12,
          fill: "none",
          stroke: "#ffb330",
          "stroke-width": 1.5,
          opacity: 0.5,
        });
        this.svg.appendChild(halo);

        const pin = el("circle", {
          cx,
          cy,
          r: 6,
          fill: "#ffb330",
          stroke: "#12151b",
          "stroke-width": 2,
        });
        this.svg.appendChild(pin);

        const label = el("text", {
          x: cx,
          y: cy - 16,
          "text-anchor": "middle",
          fill: "#ffb330",
          "font-size": 12,
          "font-weight": 600,
          "font-family": "Rajdhani, sans-serif",
        });
        label.textContent = "YOUR CAR";
        this.svg.appendChild(label);
      }
    }
  }

  window.ScatterChart = ScatterChart;
})();
