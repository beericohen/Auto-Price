(function () {
  const NS = "http://www.w3.org/2000/svg";
  const W = 900;
  const H = 460;
  const P = { t: 24, r: 28, b: 52, l: 74 }; // Padding

  // Helper: Create an SVG element with attributes
  const E = (tag, a = {}) => {
    const n = document.createElementNS(NS, tag);
    Object.entries(a).forEach(([k, v]) => n.setAttribute(k, v));
    return n;
  };

  const money = (n) => "₪" + Math.round(n).toLocaleString("en-US");

  class ScatterChart {
    constructor(svg, records) {
      this.svg = svg;
      this.records = records;
      this.axis = "mileage";
      this.you = null;
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      this.render();
    }

    setAxis(a) {
      this.axis = a;
      this.render();
    }

    setYou(p) {
      this.you = p;
      this.render();
    }

    render() {
      // Clear existing SVG contents
      while (this.svg.firstChild) {
        this.svg.removeChild(this.svg.firstChild);
      }

      const data = this.records;
      const xs = data
        .map((r) => r[this.axis])
        .concat(this.you ? [this.you[this.axis]] : []);
      const ys = data
        .map((r) => r.price)
        .concat(this.you ? [this.you.price] : []);

      // Calculate graph bounds
      const xmin = Math.min(...xs);
      const xmax = Math.max(...xs);
      const ymax = Math.max(...ys) * 1.08; // Add 8% top padding
      const iw = W - P.l - P.r;
      const ih = H - P.t - P.b;

      // Coordinate scaling functions
      const sx = (v) => P.l + ((v - xmin) / (xmax - xmin || 1)) * iw;
      const sy = (v) => P.t + ih - (v / ymax) * ih;

      // Draw Y-axis gridlines and labels
      for (let i = 0; i < 5; i++) {
        const y = P.t + ih - (i * ih) / 4;
        const l = E("line", { x1: P.l, x2: P.l + iw, y1: y, y2: y, class: "gridline" });
        this.svg.append(l);

        const t = E("text", { x: P.l - 12, y: y + 4, "text-anchor": "end", class: "axis-text" });
        t.textContent = money((ymax * i) / 4);
        this.svg.append(t);
      }

      // Draw X-axis labels
      for (let i = 0; i < 6; i++) {
        const v = xmin + ((xmax - xmin) * i) / 5;
        const x = sx(v);
        const t = E("text", { x, y: H - P.b + 25, "text-anchor": "middle", class: "axis-text" });
        t.textContent = this.axis === "mileage" ? Math.round(v / 1000) + "k" : Math.round(v);
        this.svg.append(t);
      }

      // Draw X-axis title
      const xl = E("text", { x: P.l + iw / 2, y: H - 10, "text-anchor": "middle", class: "axis-title" });
      xl.textContent = this.axis === "mileage" ? "MILEAGE (KM)" : "YEAR";
      this.svg.append(xl);

      // Plot market data points
      data.forEach((r) => {
        const c = E("circle", { cx: sx(r[this.axis]), cy: sy(r.price), r: 2.8, class: "market-point" });
        const title = E("title");
        title.textContent = `${r.manufacturer} ${r.model} · ${r.year} · ${money(r.price)}`;
        c.append(title);
        this.svg.append(c);
      });

      // Plot the user's specific car if provided
      if (this.you) {
        const x = sx(this.you[this.axis]);
        const y = sy(this.you.price);
        
        const halo = E("circle", { cx: x, cy: y, r: 15, class: "you-halo" });
        const pin = E("circle", { cx: x, cy: y, r: 6, class: "you-point" });
        const lab = E("text", { x, y: y - 20, "text-anchor": "middle", class: "you-label" });
        
        lab.textContent = "YOUR CAR";
        this.svg.append(halo, pin, lab);
      }
    }
  }
  
  window.ScatterChart = ScatterChart;
})();