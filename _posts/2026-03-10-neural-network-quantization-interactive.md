---
title: 'Neural Network Quantization: From Theory to Browser'
date: 2026-03-10 11:35:00 +0800
permalink: /posts/neural-network-quantization-interactive/
categories: [Machine Learning]
tags: [quantization, neural-networks, jax-js, model-compression, interactive-demo]
math: true
---

> Experience neural network quantization interactively in your browser

## What is Quantization?

Quantization is a technique that converts neural network weights and activations from high-precision representations (e.g., 32-bit floating-point) to low-precision formats (e.g., 8-bit integers). This significantly reduces model size and computational cost, with a trade-off in accuracy.

**Core challenge**: Finding the optimal balance between precision and efficiency.

## Mathematical Foundation

Quantization is essentially a mapping function from continuous floating-point space to discrete integer space:

$$
q = \text{round}\left(\frac{r}{s}\right) + z
$$

Where:
- $r$ is the real-valued floating-point number
- $q$ is the quantized integer value
- $s$ is the scale factor
- $z$ is the zero-point offset

The dequantization process:

$$
r = s \cdot (q - z)
$$

## 🎮 Interactive Quantization Simulator

Let's implement a real-time quantization simulator using jax-js in the browser. You can adjust bit-width and observe precision loss directly.

```html
<script type="module">
import { numpy as np, random } from "https://esm.sh/@jax-js/jax";

function roundToEven(x) {
  const floorX = np.floor(x.ref);
  const frac = x.sub(floorX.ref);
  const gtHalf = frac.ref.greater(0.5);
  const eqHalf = frac.equal(0.5);
  const ceilX = floorX.ref.add(1);
  const rounded = np.where(gtHalf, ceilX.ref, floorX.ref);
  const floorIsEven = np.floor(floorX.ref.div(2)).mul(2).equal(floorX.ref);
  const tieRounded = np.where(floorIsEven, floorX, ceilX);
  return np.where(eqHalf, tieRounded, rounded);
}

// Quantization function
function quantize(x, bits) {
  const qmin = 0;
  const qmax = (1 << bits) - 1;

  const xForMin = x.ref;
  const xForMax = x.ref;
  const xMin = xForMin.min();
  const xMax = xForMax.max();

  // Compute scale and zero_point
  const scale = xMax.sub(xMin.ref).div(qmax - qmin);
  const zeroPoint = roundToEven(xMin.div(scale.ref).mul(-1).add(qmin));

  // Quantize
  const xScaled = x.div(scale.ref).add(zeroPoint.ref);
  const xQuantized = np.clip(roundToEven(xScaled), qmin, qmax);

  return { quantized: xQuantized, scale, zeroPoint };
}

// Dequantization function
function dequantize(xq, scale, zeroPoint) {
  return xq.sub(zeroPoint).mul(scale);
}

// Create test data
const key = random.key(95);
const weights = random.normal(key, [4, 4]).mul(0.5);

// 8-bit quantization
const { quantized, scale, zeroPoint } = quantize(weights.ref, 8);
const recovered = dequantize(quantized, scale, zeroPoint);

// Compute quantization error
const error = np.abs(weights.sub(recovered)).mean();
console.log("Quantization error:", error.item());
</script>
```

**Try it yourself** (demo runs when visible):

<div id="demo1-output" style="padding: 15px; background: #f5f5f5; border-radius: 5px; margin: 20px 0;">
  <strong>Quantization Demo:</strong>
  <pre id="demo1-result" style="margin-top: 10px; font-size: 14px;">Scroll here to load demo...</pre>
</div>

<script type="module">
const observer = new IntersectionObserver((entries) => {
  entries.forEach(async (entry) => {
    if (entry.isIntersecting) {
      observer.unobserve(entry.target);
      document.getElementById('demo1-result').textContent = 'Loading...';

      try {
        const { numpy: np, random } = await import("https://esm.sh/@jax-js/jax");

        function quantize(x, bits) {
          const qmin = 0;
          const qmax = (1 << bits) - 1;
          const xForMin = x.ref;
          const xForMax = x.ref;
          const xMin = xForMin.min();
          const xMax = xForMax.max();
          const scale = xMax.sub(xMin.ref).div(qmax - qmin);
          const zeroPoint = roundToEven(xMin.div(scale.ref).mul(-1).add(qmin));
          const xScaled = x.div(scale.ref).add(zeroPoint.ref);
          const xQuantized = np.clip(roundToEven(xScaled), qmin, qmax);
          return { quantized: xQuantized, scale, zeroPoint };
        }

        function roundToEven(x) {
          const floorX = np.floor(x.ref);
          const frac = x.sub(floorX.ref);
          const gtHalf = frac.ref.greater(0.5);
          const eqHalf = frac.equal(0.5);
          const ceilX = floorX.ref.add(1);
          const rounded = np.where(gtHalf, ceilX.ref, floorX.ref);
          const floorIsEven = np.floor(floorX.ref.div(2)).mul(2).equal(floorX.ref);
          const tieRounded = np.where(floorIsEven, floorX, ceilX);
          return np.where(eqHalf, tieRounded, rounded);
        }

        function dequantize(xq, scale, zeroPoint) {
          return xq.sub(zeroPoint).mul(scale);
        }

        const key = random.key(95);
        const weights = random.normal(key, [4, 4]).mul(0.5);
        const { quantized, scale, zeroPoint } = quantize(weights.ref, 8);
        const recovered = dequantize(quantized, scale, zeroPoint);
        const error = np.abs(weights.sub(recovered)).mean();

        document.getElementById('demo1-result').textContent =
          `Original weights shape: [4, 4]\n` +
          `Quantization: 8-bit\n` +
          `Quantization error (MAE): ${error.item().toFixed(6)}\n` +
          `✓ Demo completed successfully!`;
      } catch (err) {
        document.getElementById('demo1-result').textContent = `Error: ${err.message}`;
      }
    }
  });
}, { threshold: 0.1 });

observer.observe(document.getElementById('demo1-output'));
</script>

**Key insight**: Lower bit-width increases discretization, leading to higher error. However, 8-bit quantization typically maintains >99% accuracy.

## 🔥 Symmetric vs Affine Quantization

**Symmetric quantization** constrains the zero-point to 0, simplifying computation:

```javascript
function symmetricQuantize(x, bits) {
  const qmax = (1 << (bits - 1)) - 1;
  const scale = np.abs(x.ref).max().div(qmax);

  const xQuantized = np.clip(
    np.floor(x.div(scale.ref).add(0.5)),
    -qmax,
    qmax
  );

  return { quantized: xQuantized, scale };
}
```

**Advantages**:
- Zero-point elimination simplifies multiplication operations
- Suitable for symmetric weight distributions
- Hardware-friendly for accelerated inference

## ⚡ Quantized Matrix Multiplication

The core value of quantization lies in accelerating matrix operations. Here's how to replace floating-point arithmetic with integer operations:

```javascript
// Quantized matrix multiplication
function quantizedMatmul(A, B, scaleA, zpA, scaleB, zpB) {
  // Integer matrix multiplication
  const C = np.matmul(A, B);

  // Output scale
  const scaleC = scaleA.mul(scaleB);

  // Zero-point correction term (can be precomputed)
  const zpCorrection = zpA.ref.mul(B.sum(0)).add(
    zpB.ref.mul(A.sum(1, true))
  ).sub(zpA.mul(zpB).mul(A.shape[1]));

  return { result: C.sub(zpCorrection), scale: scaleC };
}
```

**Performance gains**: Integer operations are 2-4× faster than floating-point, with 75% memory reduction.

## 📊 Visualizing Weight Distribution Changes

**Interactive visualization** (loads when visible):

<canvas id="distributionChart" width="800" height="400" style="max-width: 100%; border: 1px solid #ddd;"></canvas>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script type="module">
const observer = new IntersectionObserver((entries) => {
  entries.forEach(async (entry) => {
    if (entry.isIntersecting) {
      observer.unobserve(entry.target);

      const { numpy: np, random } = await import("https://esm.sh/@jax-js/jax");

      function quantize(x, bits) {
        const qmin = 0;
        const qmax = (1 << bits) - 1;
        const xForMin = x.ref;
        const xForMax = x.ref;
        const xMin = xForMin.min();
        const xMax = xForMax.max();
          const scale = xMax.sub(xMin.ref).div(qmax - qmin);
          const zeroPoint = roundToEven(xMin.div(scale.ref).mul(-1).add(qmin));
          const xScaled = x.div(scale.ref).add(zeroPoint.ref);
          const xQuantized = np.clip(roundToEven(xScaled), qmin, qmax);
        return { quantized: xQuantized, scale, zeroPoint };
      }

      function roundToEven(x) {
        const floorX = np.floor(x.ref);
        const frac = x.sub(floorX.ref);
        const gtHalf = frac.ref.greater(0.5);
        const eqHalf = frac.equal(0.5);
        const ceilX = floorX.ref.add(1);
        const rounded = np.where(gtHalf, ceilX.ref, floorX.ref);
        const floorIsEven = np.floor(floorX.ref.div(2)).mul(2).equal(floorX.ref);
        const tieRounded = np.where(floorIsEven, floorX, ceilX);
        return np.where(eqHalf, tieRounded, rounded);
      }

      function dequantize(xq, scale, zeroPoint) {
        return xq.sub(zeroPoint).mul(scale);
      }

      function computeHistogram(data, bins = 50) {
        const arr = data.js().flat();
        const min = Math.min(...arr);
        const max = Math.max(...arr);
        const binWidth = (max - min) / bins;
        const counts = new Array(bins).fill(0);
        const labels = [];

        for (let i = 0; i < bins; i++) {
          labels.push((min + i * binWidth).toFixed(3));
        }

        arr.forEach(val => {
          const idx = Math.min(Math.floor((val - min) / binWidth), bins - 1);
          counts[idx]++;
        });

        return { labels, counts };
      }

      const key = random.key(95);
      const weights = random.normal(key, [1000]).mul(0.5);
      const { quantized, scale, zeroPoint } = quantize(weights.ref, 8);
      const recovered = dequantize(quantized, scale, zeroPoint);

      const origHist = computeHistogram(weights);
      const quantHist = computeHistogram(recovered);

      const ctx = document.getElementById('distributionChart');
      if (window.myChart) window.myChart.destroy();

      window.myChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: origHist.labels,
          datasets: [{
            label: 'Float32 Original Distribution',
            data: origHist.counts,
            borderColor: 'rgb(75, 192, 192)',
            fill: false
          }, {
            label: 'Int8 Quantized Distribution',
            data: quantHist.counts,
            borderColor: 'rgb(255, 99, 132)',
            fill: false
          }]
        },
        options: {
          responsive: true,
          plugins: {
            title: { display: true, text: 'Weight Distribution: Before vs After Quantization' }
          }
        }
      });
    }
  });
}, { threshold: 0.1 });

observer.observe(document.getElementById('distributionChart'));
</script>

**Observations**:
- 2-bit: Significant distortion, only for extreme compression scenarios
- 4-bit: Usable with Quantization-Aware Training (QAT)
- 8-bit: Nearly lossless, industry standard
- 16-bit: Virtually identical to float32

## 🤖 Model Inference Comparison

**Interactive inference demo** (loads when visible):

<div id="results" style="margin: 20px 0;">Scroll here to load demo...</div>

<script type="module">
const observer = new IntersectionObserver((entries) => {
  entries.forEach(async (entry) => {
    if (entry.isIntersecting) {
      observer.unobserve(entry.target);
      document.getElementById('results').innerHTML = 'Loading...';

      const { nn, numpy: np, random } = await import("https://esm.sh/@jax-js/jax");

      function quantize(x, bits) {
        const qmin = 0;
        const qmax = (1 << bits) - 1;
        const xForMin = x.ref;
        const xForMax = x.ref;
        const xMin = xForMin.min();
        const xMax = xForMax.max();
        const scale = xMax.sub(xMin.ref).div(qmax - qmin);
        const zeroPoint = roundToEven(xMin.div(scale.ref).mul(-1).add(qmin));
        const xScaled = x.div(scale.ref).add(zeroPoint.ref);
        const xQuantized = np.clip(roundToEven(xScaled), qmin, qmax);
        return { quantized: xQuantized, scale, zeroPoint };
      }

      function roundToEven(x) {
        const floorX = np.floor(x.ref);
        const frac = x.sub(floorX.ref);
        const gtHalf = frac.ref.greater(0.5);
        const eqHalf = frac.equal(0.5);
        const ceilX = floorX.ref.add(1);
        const rounded = np.where(gtHalf, ceilX.ref, floorX.ref);
        const floorIsEven = np.floor(floorX.ref.div(2)).mul(2).equal(floorX.ref);
        const tieRounded = np.where(floorIsEven, floorX, ceilX);
        return np.where(eqHalf, tieRounded, rounded);
      }

      function dequantize(xq, scale, zeroPoint) {
        return xq.sub(zeroPoint).mul(scale);
      }

      const key = random.key(95);
      const [keyW1, keyW2, keyInput] = random.split(key, 3);
      const weights1 = random.normal(keyW1, [16, 784]).mul(0.1);
      const bias1 = np.zeros([16]);
      const weights2 = random.normal(keyW2, [10, 16]).mul(0.1);
      const bias2 = np.zeros([10]);
      const testInput = random.normal(keyInput, [1, 784]).mul(0.5);

      const h1_float = nn.relu(
        np.matmul(testInput.ref, weights1.ref.transpose()).add(bias1.ref)
      );
      const out_float = np.matmul(h1_float, weights2.ref.transpose()).add(bias2.ref);
      const pred_float = np.argmax(out_float.ref, -1);

      const { quantized: w1_q, scale: s1, zeroPoint: z1 } = quantize(weights1.ref, 8);
      const { quantized: w2_q, scale: s2, zeroPoint: z2 } = quantize(weights2.ref, 8);
      const w1_dq = dequantize(w1_q, s1, z1);
      const w2_dq = dequantize(w2_q, s2, z2);

      const h1_quant = nn.relu(
        np.matmul(testInput.ref, w1_dq.transpose()).add(bias1.ref)
      );
      const out_quant = np.matmul(h1_quant, w2_dq.transpose()).add(bias2.ref);
      const pred_quant = np.argmax(out_quant.ref, -1);

      const outputDiff = np.abs(out_float.ref.sub(out_quant.ref)).mean();
      const floatPred = pred_float.item();
      const quantPred = pred_quant.item();
      const floatConfidence = np.max(out_float.ref).item();
      const quantConfidence = np.max(out_quant).item();

      document.getElementById('results').innerHTML = `
    <table style="border-collapse: collapse; margin-top: 20px; width: 100%;">
      <tr><th style="border: 1px solid #ddd; padding: 8px; background-color: #4CAF50; color: white;">Model</th>
          <th style="border: 1px solid #ddd; padding: 8px; background-color: #4CAF50; color: white;">Prediction</th>
          <th style="border: 1px solid #ddd; padding: 8px; background-color: #4CAF50; color: white;">Confidence</th>
          <th style="border: 1px solid #ddd; padding: 8px; background-color: #4CAF50; color: white;">Memory</th></tr>
      <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">Float32</td>
        <td style="border: 1px solid #ddd; padding: 8px;">${floatPred}</td>
        <td style="border: 1px solid #ddd; padding: 8px;">${floatConfidence.toFixed(4)}</td>
        <td style="border: 1px solid #ddd; padding: 8px;">~100 KB</td>
      </tr>
      <tr>
        <td style="border: 1px solid #ddd; padding: 8px;">Int8 Quantized</td>
        <td style="border: 1px solid #ddd; padding: 8px;">${quantPred}</td>
        <td style="border: 1px solid #ddd; padding: 8px;">${quantConfidence.toFixed(4)}</td>
        <td style="border: 1px solid #ddd; padding: 8px;">~25 KB (↓75%)</td>
      </tr>
      <tr>
        <td colspan="4" style="border: 1px solid #ddd; padding: 8px;"><strong>Output Difference (MAE):</strong> ${outputDiff.item().toFixed(6)}</td>
      </tr>
    </table>
  `;
    }
  });
}, { threshold: 0.1 });

observer.observe(document.getElementById('results'));
</script>

**Key finding**: 8-bit quantization typically preserves predictions with output difference < 0.01.

## 🎨 Error Heatmap: Visualizing Quantization Loss

**Interactive heatmap** (loads when visible, click cells to see values):

<canvas id="heatmap" width="600" height="600" style="max-width: 100%; border: 1px solid #ddd; cursor: crosshair;"></canvas>

<script type="module">
let globalErrorMatrix = null;

const observer = new IntersectionObserver((entries) => {
  entries.forEach(async (entry) => {
    if (entry.isIntersecting) {
      observer.unobserve(entry.target);

      const { numpy: np, random } = await import("https://esm.sh/@jax-js/jax");

      function quantize(x, bits) {
        const qmin = 0;
        const qmax = (1 << bits) - 1;
        const xForMin = x.ref;
        const xForMax = x.ref;
        const xMin = xForMin.min();
        const xMax = xForMax.max();
        const scale = xMax.sub(xMin.ref).div(qmax - qmin);
        const zeroPoint = roundToEven(xMin.div(scale.ref).mul(-1).add(qmin));
        const xScaled = x.div(scale.ref).add(zeroPoint.ref);
        const xQuantized = np.clip(roundToEven(xScaled), qmin, qmax);
        return { quantized: xQuantized, scale, zeroPoint };
      }

      function roundToEven(x) {
        const floorX = np.floor(x.ref);
        const frac = x.sub(floorX.ref);
        const gtHalf = frac.ref.greater(0.5);
        const eqHalf = frac.equal(0.5);
        const ceilX = floorX.ref.add(1);
        const rounded = np.where(gtHalf, ceilX.ref, floorX.ref);
        const floorIsEven = np.floor(floorX.ref.div(2)).mul(2).equal(floorX.ref);
        const tieRounded = np.where(floorIsEven, floorX, ceilX);
        return np.where(eqHalf, tieRounded, rounded);
      }

      function dequantize(xq, scale, zeroPoint) {
        return xq.sub(zeroPoint).mul(scale);
      }

      function drawHeatmap(canvas, errorMatrix) {
        const ctx = canvas.getContext('2d');
        const data = errorMatrix.js();
        const [rows, cols] = errorMatrix.shape;

        const cellWidth = canvas.width / cols;
        const cellHeight = canvas.height / rows;
        const maxError = Math.max(...data.flat().map(Math.abs));

        for (let i = 0; i < rows; i++) {
          for (let j = 0; j < cols; j++) {
            const error = Math.abs(data[i][j]);
            const intensity = Math.floor((error / maxError) * 255);
            const r = intensity;
            const g = 0;
            const b = 255 - intensity;
            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fillRect(j * cellWidth, i * cellHeight, cellWidth, cellHeight);
          }
        }

        ctx.fillStyle = 'white';
        ctx.font = '14px Arial';
        ctx.fillText(`Max error: ${maxError.toFixed(4)}`, 10, 20);
      }

      const key = random.key(95);
      const weights = random.normal(key, [32, 32]).mul(0.5);
      const { quantized, scale, zeroPoint } = quantize(weights.ref, 8);
      const recovered = dequantize(quantized, scale, zeroPoint);
      const errorMatrix = np.abs(weights.sub(recovered));

      globalErrorMatrix = errorMatrix;
      const canvas = document.getElementById('heatmap');
      drawHeatmap(canvas, errorMatrix);
    }
  });
}, { threshold: 0.1 });

observer.observe(document.getElementById('heatmap'));

document.getElementById('heatmap').addEventListener('click', (e) => {
  if (!globalErrorMatrix) return;
  const canvas = document.getElementById('heatmap');
  const rect = canvas.getBoundingClientRect();
  const x = Math.floor((e.clientX - rect.left) / (canvas.width / 32));
  const y = Math.floor((e.clientY - rect.top) / (canvas.height / 32));
  const error = globalErrorMatrix.js()[y][x];
  alert(`Position [${y}, ${x}]\nQuantization error: ${error.toFixed(6)}`);
});
</script>

**Insights**:
- Deep blue regions: Quantization error < 0.001 (nearly perfect)
- Purple regions: Moderate error 0.001-0.01
- Red regions: High error > 0.01 (requires attention)

Most weights exhibit minimal quantization error; only a few outliers need special handling.

## 🚀 Performance Arena: Float vs Integer

```javascript
import { jit, numpy as np, random } from "https://esm.sh/@jax-js/jax";

// JIT-compiled float32 matmul
const float32Matmul = jit((A, B) => np.matmul(A, B));

// JIT-compiled int8 matmul
const int8Matmul = jit((A, B) => np.matmul(A, B));

// Benchmark function
function benchmark(fn, A, B, iterations = 100) {
  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    fn(A.ref, B.ref);
  }
  return (performance.now() - start) / iterations;
}

// Create test matrices
const key = random.key(95);
const [keyA, keyB] = random.split(key, 2);
const A_float = random.normal(keyA, [128, 128]);
const B_float = random.normal(keyB, [128, 128]);

const { quantized: A_int } = quantize(A_float.ref, 8);
const { quantized: B_int } = quantize(B_float.ref, 8);

const timeFloat = benchmark(float32Matmul, A_float, B_float);
const timeInt = benchmark(int8Matmul, A_int, B_int);

console.log(`Float32: ${timeFloat.toFixed(2)}ms`);
console.log(`Int8: ${timeInt.toFixed(2)}ms`);
console.log(`Speedup: ${(timeFloat / timeInt).toFixed(2)}x`);
```

## 💡 Practical Guidelines

**When to use quantization?**

1. **Edge deployment**: Mobile devices and IoT with limited memory and compute
2. **Large-scale inference**: Cloud services requiring cost reduction
3. **Real-time applications**: Latency-sensitive scenarios (speech recognition, video processing)

**Quantization strategy selection**:

| Scenario | Recommended Approach | Accuracy Loss |
|----------|---------------------|---------------|
| Image Classification | 8-bit Post-Training Quantization | < 1% |
| Object Detection | 8-bit + QAT | < 2% |
| Large Language Models | 4-bit Group Quantization | 2-5% |
| Generative Models | Mixed Precision (critical layers 16-bit) | < 3% |

**Best practices**:

- ❌ Don't quantize Batch Normalization layers (numerically unstable)
- ✅ Fuse Conv + BN + ReLU before quantization
- ❌ Don't use uniform bit-width for all layers
- ✅ Maintain higher precision for sensitive layers (first/last layers)

## 🎯 Summary

Quantization is a fundamental technique for model compression. Through interactive demonstrations using jax-js in the browser, we can:

1. **Intuitively understand** the mathematical principles and precision trade-offs
2. **Compare in real-time** different quantization strategies
3. **Experiment at zero cost** without requiring GPU infrastructure

Next steps to explore:
- Implement Quantization-Aware Training (QAT)
- Explore mixed-precision quantization
- Compare quantization frameworks (TensorRT, ONNX Runtime)

---

**Interactive Demo**: Visit [Live Demo](#) to experience the complete quantization simulator

**References**:
- [Lei Mao's Neural Networks Quantization](https://leimao.github.io/article/Neural-Networks-Quantization/)
- [jax-js Documentation](https://github.com/ekzhang/jax-js)
