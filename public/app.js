const healthOut = document.getElementById('healthOut');
const exportOut = document.getElementById('exportOut');
const btnHealth = document.getElementById('btnHealth');
const btnCreate = document.getElementById('btnCreate');
const btnPoll = document.getElementById('btnPoll');

const userEmail = document.getElementById('userEmail');
const productName = document.getElementById('productName');
const exportFormat = document.getElementById('exportFormat');

let lastTaskId = null;

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

async function safeJson(res) {
  const text = await res.text();
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text };
  }
}

btnHealth.addEventListener('click', async () => {
  healthOut.textContent = 'loading...';
  try {
    const res = await fetch('/api/health');
    const data = await safeJson(res);
    healthOut.textContent = pretty({ status: res.status, data });
  } catch (e) {
    healthOut.textContent = String(e);
  }
});

btnCreate.addEventListener('click', async () => {
  exportOut.textContent = 'loading...';
  btnPoll.disabled = true;
  lastTaskId = null;
  try {
    const payload = {
      exportFormat: exportFormat.value,
      userEmail: userEmail.value,
      productName: productName.value,
      languageDefault: 'zh',
      cards: [{ name: 'PV', total: 1, average: 1 }],
      charts: []
    };
    const res = await fetch('/api/v1/export', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await safeJson(res);
    exportOut.textContent = pretty({ status: res.status, data });
    if (res.ok && data && data.taskId) {
      lastTaskId = data.taskId;
      btnPoll.disabled = false;
    }
  } catch (e) {
    exportOut.textContent = String(e);
  }
});

btnPoll.addEventListener('click', async () => {
  if (!lastTaskId) return;
  exportOut.textContent = 'loading...';
  try {
    const qs = new URLSearchParams({ taskId: lastTaskId, userEmail: userEmail.value });
    const res = await fetch(`/api/v1/export/status?${qs.toString()}`);
    const data = await safeJson(res);
    exportOut.textContent = pretty({ status: res.status, data });
  } catch (e) {
    exportOut.textContent = String(e);
  }
});
