// Preset Data for one-click testing
const PRESETS = {
  healthy: {
    age: 38, sex: "0", cp: "0", trestbps: 118, chol: 180,
    fbs: "0", restecg: "0", thalach: 175, exang: "0", oldpeak: 0.2,
    slope: "2", ca: "0", thal: "1"
  },
  moderate: {
    age: 54, sex: "1", cp: "1", trestbps: 135, chol: 235,
    fbs: "0", restecg: "1", thalach: 148, exang: "0", oldpeak: 1.2,
    slope: "1", ca: "1", thal: "2"
  },
  high: {
    age: 65, sex: "1", cp: "3", trestbps: 160, chol: 286,
    fbs: "1", restecg: "2", thalach: 108, exang: "1", oldpeak: 3.6,
    slope: "1", ca: "2", thal: "3"
  }
};

function loadPreset(key) {
  const data = PRESETS[key];
  if (!data) return;

  // Active state
  document.querySelectorAll('.btn-preset').forEach(b => b.classList.remove('active'));
  const activeBtn = document.getElementById('preset-' + key);
  if (activeBtn) activeBtn.classList.add('active');

  // Fill all 13 inputs
  for (const [field, val] of Object.entries(data)) {
    const el = document.getElementById('input-' + field);
    if (el) el.value = val;
  }
}

// Prediction form submit handler
document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('prediction-form');
  const submitBtn = document.getElementById('btn-submit-prediction');
  const btnLabel = document.getElementById('btn-label');
  const errorCard = document.getElementById('error-card');
  const errorText = document.getElementById('error-text');
  const resultsContainer = document.getElementById('results-container');

  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    errorCard.style.display = 'none';
    submitBtn.disabled = true;
    btnLabel.textContent = 'Analyzing Diagnostic Indicators...';

    // Collect 13 parameters
    const formData = new FormData(form);
    const payload = {};
    formData.forEach((val, key) => { payload[key] = parseFloat(val); });

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorJson = await response.json().catch(() => ({}));
        throw new Error(errorJson.error || `Server responded with HTTP ${response.status}`);
      }

      const res = await response.json();
      if (!res.success) {
        throw new Error(res.error || 'Prediction calculation failed.');
      }

      const data = res.data;
      const badgeClass = data.badge_class || (data.risk_percentage < 35 ? 'success' : data.risk_percentage < 65 ? 'warning' : 'danger');

      resultsContainer.innerHTML = `
        <section id="results-section" class="result-card risk-${badgeClass}">
          <div class="result-header">
            <div>
              <span class="result-badge badge-${badgeClass}">
                ${data.icon} ${data.risk_level}
              </span>
            </div>
            <div style="text-align: right;">
              <div style="font-size: 0.85rem; color: var(--text-muted);">Estimated Disease Risk Probability</div>
              <div style="font-size: 1.8rem; font-weight: 800; color: #ffffff;">
                ${data.risk_percentage}%
              </div>
            </div>
          </div>

          <div class="prob-meter-wrapper">
            <div class="prob-meter-labels">
              <span>0% (Low Risk)</span>
              <span>50% (Threshold)</span>
              <span>100% (High Risk)</span>
            </div>
            <div class="prob-bar-container">
              <div class="prob-bar-fill fill-${badgeClass}" style="width: ${data.risk_percentage}%;"></div>
            </div>
          </div>

          <p style="color: var(--text-main); font-size: 1rem; margin-bottom: 12px;">
            ${data.summary}
          </p>

          <div class="factors-box">
            <div class="factors-title">Patient Profile Clinical Drivers:</div>
            <ul class="factors-list">
              ${data.risk_factors.map(f => `<li>${f}</li>`).join('')}
            </ul>
          </div>
        </section>
      `;

      const resultsEl = document.getElementById('results-section');
      if (resultsEl) {
        resultsEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }

    } catch (err) {
      errorText.textContent = err.message || 'Network error occurred while contacting prediction API.';
      errorCard.style.display = 'block';
    } finally {
      submitBtn.disabled = false;
      btnLabel.textContent = 'Run Cardiovascular Analysis';
    }
  });
});
