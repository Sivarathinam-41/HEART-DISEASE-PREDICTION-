// CardioSense AI — Core Application Logic & Chat Assistant

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

let currentPatientData = null;
let currentRiskResult = null;

// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll('.nav-tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

  const btn = document.getElementById('btn-tab-' + tabId);
  const pane = document.getElementById('tab-' + tabId);

  if (btn) btn.classList.add('active');
  if (pane) pane.classList.add('active');
}

// Preset Loader
function loadPreset(key) {
  const data = PRESETS[key];
  if (!data) return;

  document.querySelectorAll('.btn-preset').forEach(b => b.classList.remove('active'));
  const activeBtn = document.getElementById('preset-' + key);
  if (activeBtn) activeBtn.classList.add('active');

  for (const [field, val] of Object.entries(data)) {
    const el = document.getElementById('input-' + field);
    if (el) el.value = val;
  }
}

// Render Recommendations in Tab 2
function populateRecommendations(recs, riskResult) {
  const badgeEl = document.getElementById('rec-risk-badge');
  if (badgeEl && riskResult) {
    badgeEl.textContent = `${riskResult.icon} ${riskResult.risk_level} (${riskResult.risk_percentage}%)`;
    badgeEl.className = `result-badge badge-${riskResult.badge_class}`;
  }

  // Diet
  const dietList = document.getElementById('rec-diet-list');
  if (dietList && recs.diet) {
    dietList.innerHTML = recs.diet.map(item => `<li>${item}</li>`).join('');
  }

  // Exercise
  const exerciseList = document.getElementById('rec-exercise-list');
  if (exerciseList && recs.exercise) {
    exerciseList.innerHTML = recs.exercise.map(item => `<li>${item}</li>`).join('');
  }

  // Medical Actions
  const medicalList = document.getElementById('rec-medical-list');
  if (medicalList && recs.medical_actions) {
    medicalList.innerHTML = recs.medical_actions.map(item => `<li>${item}</li>`).join('');
  }

  // Biomarker Table
  const tableBody = document.getElementById('biomarker-table-body');
  if (tableBody && recs.biomarkers) {
    tableBody.innerHTML = recs.biomarkers.map(b => {
      let statusClass = 'status-normal';
      if (b.status === 'Elevated' || b.status === 'Borderline') statusClass = 'status-elevated';
      if (b.status === 'High' || b.status === 'Ischemia Risk') statusClass = 'status-high';
      return `
        <tr>
          <td><strong>${b.metric}</strong></td>
          <td>${b.current}</td>
          <td>${b.target}</td>
          <td><span class="badge-status ${statusClass}">${b.status}</span></td>
        </tr>
      `;
    }).join('');
  }
}

// Send Message to CardioBot
async function sendChatMessage(presetMsg) {
  const inputEl = document.getElementById('chat-input-field');
  const chatMessages = document.getElementById('chat-messages');

  const text = presetMsg || inputEl.value.trim();
  if (!text) return;

  if (!presetMsg) inputEl.value = '';

  // Append user message
  const userMsgEl = document.createElement('div');
  userMsgEl.className = 'chat-msg msg-user';
  userMsgEl.textContent = text;
  chatMessages.appendChild(userMsgEl);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  // Append typing placeholder
  const botMsgEl = document.createElement('div');
  botMsgEl.className = 'chat-msg msg-bot';
  botMsgEl.textContent = 'Analyzing medical query...';
  chatMessages.appendChild(botMsgEl);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        patient_data: currentPatientData,
        risk_result: currentRiskResult
      })
    });

    const data = await res.json();
    if (data.success) {
      botMsgEl.innerHTML = data.response
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');
    } else {
      botMsgEl.textContent = data.error || 'Unable to process query.';
    }
  } catch (err) {
    botMsgEl.textContent = 'Network error contacting CardioBot assistant.';
  }

  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// DOM Setup
document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('prediction-form');
  const submitBtn = document.getElementById('btn-submit-prediction');
  const btnLabel = document.getElementById('btn-label');
  const errorCard = document.getElementById('error-card');
  const errorText = document.getElementById('error-text');
  const resultsContainer = document.getElementById('results-container');

  // Form submit handler
  if (form) {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();

      if (errorCard) errorCard.style.display = 'none';
      submitBtn.disabled = true;
      btnLabel.textContent = 'Analyzing Diagnostic Indicators...';

      const formData = new FormData(form);
      const payload = {};
      formData.forEach((val, key) => { payload[key] = parseFloat(val); });
      currentPatientData = payload;

      try {
        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.error || `Server returned HTTP ${response.status}`);
        }

        const res = await response.json();
        if (!res.success) {
          throw new Error(res.error || 'Prediction calculation failed.');
        }

        const data = res.data;
        currentRiskResult = data;
        const badgeClass = data.badge_class || (data.risk_percentage < 35 ? 'success' : data.risk_percentage < 65 ? 'warning' : 'danger');

        // Render Results Card
        resultsContainer.innerHTML = `
          <section id="results-section" class="result-card risk-${badgeClass}">
            <div class="result-header">
              <div>
                <span class="result-badge badge-${badgeClass}">
                  ${data.icon} ${data.risk_level}
                </span>
                <span style="font-size:0.8rem; margin-left:10px; color:var(--text-muted);">
                  <span style="color:#10b981;">●</span> Verified Clinical Assessment
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

            <div style="margin-top:20px; text-align:right;">
              <button type="button" class="btn-preset" onclick="switchTab('recommendations')">
                💡 View Personalized AI Health Recommendations →
              </button>
            </div>
          </section>
        `;

        // Automatically populate recommendations tab
        if (data.recommendations) {
          populateRecommendations(data.recommendations, data);
        }

        const resultsEl = document.getElementById('results-section');
        if (resultsEl) {
          resultsEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }

      } catch (err) {
        if (errorText) errorText.textContent = err.message || 'Error communicating with prediction server.';
        if (errorCard) errorCard.style.display = 'block';
      } finally {
        submitBtn.disabled = false;
        btnLabel.textContent = 'Run Cardiovascular Analysis';
      }
    });
  }

  // Chat input Enter key
  const chatInput = document.getElementById('chat-input-field');
  if (chatInput) {
    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        sendChatMessage();
      }
    });
  }
});
