const state = {
  reports: [],
  config: null,
  selected: null,
};

const $ = (id) => document.getElementById(id);

function fmtPrice(value) {
  if (value === null || value === undefined) return "-";
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  if (Math.abs(n) >= 100) return n.toLocaleString("zh-TW", { maximumFractionDigits: 2 });
  if (Math.abs(n) >= 1) return n.toLocaleString("zh-TW", { maximumFractionDigits: 4 });
  return n.toLocaleString("zh-TW", { maximumFractionDigits: 8 });
}

function fmtVolume(value) {
  const n = Number(value || 0);
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(2)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toFixed(0);
}

function directionLabel(value) {
  if (value === "long") return "看多";
  if (value === "short") return "看空";
  return "觀望";
}

function sideClass(value) {
  if (value === "long") return "long";
  if (value === "short") return "short";
  return "neutral";
}

function actionClass(value) {
  if (value === "market") return "market";
  if (value === "limit") return "limit";
  if (value === "watch") return "watch";
  return "avoid";
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function loadState() {
  const data = await api("/api/state");
  state.reports = data.reports || [];
  state.config = data.config;
  renderSettings(data.config);
  renderProviders(data.providers || [], data.config);
  renderMetrics(data);
  renderRows();
  if (!state.selected && state.reports.length) showDetail(state.reports[0]);
}

function renderSettings(config) {
  if (!config) return;
  $("exchange").value = config.scan.exchange;
  $("top").value = config.scan.top;
  $("minVolume").value = config.scan.min_volume;
  $("passingScore").value = config.scan.passing_score;
  $("refreshMinutes").value = config.server.refresh_minutes;
  $("symbols").value = config.scan.symbols || "";
  $("paidEnabled").checked = Boolean(config.paid_data.enabled);
  $("preferredDerivativesExchange").value = config.paid_data.preferred_derivatives_exchange || "Bybit";
}

function renderMetrics(data) {
  const reports = data.reports || [];
  const passed = reports.filter((item) => item.passed).length;
  $("metricCount").textContent = reports.length;
  $("metricPassed").textContent = passed;
  $("metricPassScore").textContent = data.passing_score;
  $("metricExchange").textContent = data.meta?.exchange || "-";
  $("metricStatus").textContent = data.running ? "刷新中" : "待命";
  $("metricDataGaps").textContent = (data.providers || []).filter((provider) => {
    const text = `${provider.state || ""}`;
    return text.includes("未設定") || text.includes("讀不到") || text.includes("失敗");
  }).length;
  $("lastUpdated").textContent = data.last_completed ? `最後刷新：${new Date(data.last_completed).toLocaleString("zh-TW")}` : "尚未完成首次刷新";
  updateCountdown(data.next_refresh_ts, data.server_time);
}

function updateCountdown(nextTs, serverTs) {
  const localNow = Date.now() / 1000;
  const drift = localNow - Number(serverTs || localNow);
  const remain = Math.max(0, Number(nextTs || 0) + drift - localNow);
  const minutes = Math.floor(remain / 60);
  const seconds = Math.floor(remain % 60);
  $("metricCountdown").textContent = `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function renderProviders(providers, config) {
  const list = $("apiList");
  list.innerHTML = "";
  const keyState = config?.api_keys || {};
  providers.forEach((provider) => {
    const card = document.createElement("article");
    card.className = "api-card";
    const configured = keyState[provider.id]?.configured || provider.configured;
    const badgeClass = providerBadgeClass(provider);
    const needsKey = provider.key_hint !== "不需要 API key";
    const lastError = provider.last_error ? `<p class="api-error">最後錯誤：${provider.last_error}</p>` : "";
    card.innerHTML = `
      <header>
        <div>
          <strong>${provider.name}</strong>
          <p>${provider.category}</p>
        </div>
        <span class="badge ${badgeClass}">${provider.state || (configured ? "已設定" : "未設定")}</span>
      </header>
      <p>${provider.purpose}</p>
      <div class="provider-health">
        <div><span>即時性</span><b>${provider.realtime_status || "未標示"}</b></div>
        <div><span>資料延遲</span><b>${provider.data_latency || "依端點而定"}</b></div>
        <div><span>本次讀取</span><b>${provider.success_count || 0} 成功 / ${provider.failure_count || 0} 失敗</b></div>
      </div>
      <p>${provider.health_note || ""}</p>
      ${needsKey ? `<label>${provider.key_hint}
        <input data-api-key="${provider.id}" type="password" placeholder="${configured ? "已設定，留空代表保留" : "貼上 API key"}">
      </label>` : `<p>此來源不需要 API key，系統會自動使用。</p>`}
      <div class="api-links">
        <a href="${provider.apply_url}" target="_blank">申請 API</a>
        <a href="${provider.docs_url}" target="_blank">官方文件</a>
      </div>
      <p>${provider.message || ""}</p>
      ${lastError}
    `;
    list.appendChild(card);
  });
}

function providerBadgeClass(provider) {
  const stateText = `${provider.state || ""}`;
  if (stateText.includes("成功") || stateText.includes("可讀取") || stateText.includes("免費內建")) return "pass";
  if (stateText.includes("部分")) return "watch";
  if (stateText.includes("失敗") || stateText.includes("讀不到")) return "fail";
  if (stateText.includes("未設定")) return "off";
  return provider.configured ? "api" : "off";
}

function renderRows() {
  const body = $("rows");
  const query = $("search").value.trim().toUpperCase();
  const filter = $("filter").value;
  body.innerHTML = "";
  const rows = state.reports.filter((item) => {
    if (query && !item.symbol.includes(query)) return false;
    if (filter === "passed" && !item.passed) return false;
    if (filter === "failed" && item.passed) return false;
    if (["long", "short", "neutral"].includes(filter) && item.direction !== filter) return false;
    if (filter.startsWith("action_") && item.trade_action !== filter.replace("action_", "")) return false;
    return true;
  });
  rows.forEach((item) => {
    const tr = document.createElement("tr");
    const entry = item.entry_zone ? `${fmtPrice(item.entry_zone[0])}-${fmtPrice(item.entry_zone[1])}` : "-";
    tr.innerHTML = `
      <td>${item.rank}</td>
      <td><strong>${item.symbol}</strong></td>
      <td><span class="badge ${sideClass(item.direction)}">${directionLabel(item.direction)}</span></td>
      <td><span class="badge ${actionClass(item.trade_action)}">${item.trade_action_label || "-"}</span></td>
      <td><span class="badge ${item.passed ? "pass" : "fail"}">${item.passed ? "及格" : "未及格"}</span></td>
      <td><span class="score">${Number(item.score).toFixed(1)}</span></td>
      <td>${Number(item.data_completeness || 0).toFixed(0)}%</td>
      <td>${item.grade}</td>
      <td>${fmtPrice(item.price)}</td>
      <td>${Number(item.change_pct_24h).toFixed(2)}%</td>
      <td>${fmtVolume(item.quote_volume_24h)}</td>
      <td>${entry}</td>
      <td>${fmtPrice(item.stop)}</td>
      <td>${fmtPrice(item.target)}</td>
      <td>${item.rr === null || item.rr === undefined ? "-" : Number(item.rr).toFixed(2)}</td>
    `;
    tr.addEventListener("click", () => showDetail(item));
    body.appendChild(tr);
  });
}

function showDetail(item) {
  state.selected = item;
  $("detailTitle").textContent = `${item.symbol} · ${directionLabel(item.direction)} · ${Number(item.score).toFixed(1)} 分`;
  $("detailMeta").textContent = `${item.trade_action_label || "未分類"} · ${item.trade_action_reason || ""} · 資料完整度 ${Number(item.data_completeness || 0).toFixed(0)}% · 可用分母 ${Number(item.available_score_max || 0).toFixed(1)}`;
  renderList("detailReasons", item.selected_reasons || item.reasons || ["目前沒有足夠共振"]);
  renderList("detailWarnings", item.selected_warnings || item.warnings || ["無重大提醒"]);
  renderRiskPlan(item);
  const features = $("detailFeatures");
  features.innerHTML = "";
  Object.entries(item.feature_scores || {}).forEach(([name, value]) => {
    const row = document.createElement("div");
    row.className = "feature";
    const maxValue = Number(item.feature_max_scores?.[name] || 14);
    const pct = Math.max(0, Math.min(100, Number(value) / Math.max(maxValue, 1) * 100));
    row.innerHTML = `<span>${featureLabel(name)}</span><div><i style="width:${pct}%"></i></div><b>${Number(value).toFixed(1)}/${maxValue.toFixed(0)}</b>`;
    features.appendChild(row);
  });
  Object.entries(item.skipped_features || {}).forEach(([name, reason]) => {
    const row = document.createElement("div");
    row.className = "feature";
    row.innerHTML = `<span>${featureLabel(name)}</span><div><i style="width:0%"></i></div><b>跳過</b>`;
    row.title = reason;
    features.appendChild(row);
  });
  $("detailPaid").textContent = JSON.stringify(item.metadata?.paid_data || { 狀態: "尚無付費資料" }, null, 2);
}

function renderRiskPlan(item) {
  const values = [];
  if (item.entry_zone) values.push(`入場區：${fmtPrice(item.entry_zone[0])} - ${fmtPrice(item.entry_zone[1])}`);
  if (item.stop !== null && item.stop !== undefined) values.push(`建議止損：${fmtPrice(item.stop)}`);
  (item.take_profits || []).forEach((tp) => {
    values.push(`${tp.name}：${fmtPrice(tp.price)}，約 ${Number(tp.rr || 0).toFixed(2)}R，建議 ${Number(tp.portion_pct || 0).toFixed(0)}% 倉位。${tp.note || ""}`);
  });
  if (item.rr !== null && item.rr !== undefined) values.push(`主目標風報比：約 ${Number(item.rr).toFixed(2)}R`);
  renderList("detailRiskPlan", values.length ? values : ["資料不足，暫無法計算止盈止損"]);
}

function renderList(id, values) {
  const node = $(id);
  node.innerHTML = "";
  (values.length ? values : ["無"]).forEach((value) => {
    const li = document.createElement("li");
    li.textContent = value;
    node.appendChild(li);
  });
}

function featureLabel(name) {
  const labels = {
    liquidity_sweep: "流動性掃蕩",
    htf_poi: "高週期 POI",
    mss_bos: "MSS/BOS",
    displacement: "位移",
    fvg: "FVG",
    ote: "OTE",
    trendline: "趨勢線",
    amd: "AMD",
    nexus: "Nexus",
    risk_reward: "風報比",
    market_quality: "市場品質",
    paid_data: "付費資料",
  };
  return labels[name] || name;
}

async function saveSettings() {
  const apiKeys = {};
  document.querySelectorAll("[data-api-key]").forEach((input) => {
    const key = input.getAttribute("data-api-key");
    const configured = state.config?.api_keys?.[key]?.configured;
    apiKeys[key] = input.value ? input.value : (configured ? "__keep__" : "");
  });
  await api("/api/settings", {
    method: "POST",
    body: JSON.stringify({
      server: {
        refresh_minutes: Number($("refreshMinutes").value),
      },
      scan: {
        exchange: $("exchange").value,
        top: Number($("top").value),
        min_volume: Number($("minVolume").value),
        passing_score: Number($("passingScore").value),
        symbols: $("symbols").value.trim(),
      },
      paid_data: {
        enabled: $("paidEnabled").checked,
        preferred_derivatives_exchange: $("preferredDerivativesExchange").value,
      },
      api_keys: apiKeys,
    }),
  });
  await loadState();
}

async function scanNow() {
  $("metricStatus").textContent = "刷新中";
  await api("/api/scan", { method: "POST", body: "{}" });
  setTimeout(loadState, 1000);
}

async function checkApis() {
  $("metricStatus").textContent = "測試 API";
  const data = await api("/api/check-apis", { method: "POST", body: "{}" });
  renderProviders(data.providers || [], state.config);
  $("metricStatus").textContent = "待命";
}

$("saveSettings").addEventListener("click", saveSettings);
$("scanNow").addEventListener("click", scanNow);
$("checkApis").addEventListener("click", checkApis);
$("search").addEventListener("input", renderRows);
$("filter").addEventListener("change", renderRows);

loadState();
setInterval(loadState, 15000);
