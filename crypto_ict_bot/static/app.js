const state = {
  reports: [],
  filteredReports: [],
  standoutAlerts: [],
  config: null,
  selected: null,
  loadingState: false,
  loadingHeartbeat: false,
  nextStateLoadAt: 0,
  stateVersion: null,
  runtime: {
    running: false,
    lastStarted: null,
    lastCompleted: null,
    serverTime: 0,
    receivedAt: 0,
  },
  countdown: {
    nextTs: 0,
    serverTs: 0,
    receivedAt: 0,
  },
  rows: {
    frame: null,
    scrollFrame: null,
    resetScroll: false,
  },
};

const $ = (id) => document.getElementById(id);
const TABLE_COLUMN_COUNT = 21;
const ROW_HEIGHT = 54;
const ROW_OVERSCAN = 8;
const VIRTUALIZE_AFTER = 240;
const HEARTBEAT_IDLE_MS = 5000;
const HEARTBEAT_RUNNING_MS = 1600;
let heartbeatTimer = null;
let searchTimer = null;
window.__ictDashboardState = state;

function escapeHtml(value) {
  return String(value ?? "-").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[char]));
}

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

function executionClass(value) {
  if (value === "可以做") return "execute";
  if (value === "待確認") return "pending";
  if (value === "觀察") return "watch";
  return "avoid";
}

async function api(path, options = {}) {
  const { timeoutMs = 12000, ...fetchOptions } = options;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(path, {
      cache: "no-store",
      ...fetchOptions,
      headers: { "Content-Type": "application/json", ...(fetchOptions.headers || {}) },
      signal: fetchOptions.signal || controller.signal,
    });
    if (!response.ok) throw new Error(await response.text());
    return response.json();
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("連線逾時，暫時保留上一輪資料");
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

async function loadState() {
  if (state.loadingState) return;
  state.loadingState = true;
  state.nextStateLoadAt = Date.now() + 4000;
  if (!state.reports.length) {
    $("rows").innerHTML = `<tr><td class="empty-row" colspan="${TABLE_COLUMN_COUNT}">讀取榜單中...</td></tr>`;
  }
  try {
    const data = await api("/api/state", { timeoutMs: 20000 });
    const previousSymbol = state.selected?.symbol;
    state.reports = data.reports || [];
    state.standoutAlerts = data.standout_alerts || data.meta?.standout_alerts || [];
    state.config = data.config;
    state.stateVersion = Number(data.state_version || 0);
    renderSettings(data.config);
    renderProviders(data.providers || [], data.config);
    renderMetrics(data);
    renderStandoutAlerts(state.standoutAlerts);
    renderRows({ resetScroll: true });
    if (state.reports.length) {
      const selected = previousSymbol
        ? state.reports.find((item) => item.symbol === previousSymbol)
        : state.filteredReports[0] || state.reports[0];
      if (selected) showDetail(selected);
    }
  } catch (error) {
    $("metricStatus").textContent = "資料同步中";
    const message = error && error.message ? error.message : "後台掃描中，暫時保留上一輪榜單";
    if (state.reports.length || state.config) {
      $("lastUpdated").textContent = message;
    }
    state.nextStateLoadAt = Date.now() + 8000;
    tickCountdown();
  } finally {
    state.loadingState = false;
  }
}

async function loadHeartbeat() {
  if (state.loadingHeartbeat) return;
  state.loadingHeartbeat = true;
  try {
    const data = await api("/api/heartbeat", { timeoutMs: 3500 });
    renderMetrics(data);
    if (data.standout_alerts) renderStandoutAlerts(data.standout_alerts);
    const version = Number(data.state_version || 0);
    const shouldLoadState = state.stateVersion === null || version !== state.stateVersion;
    if (shouldLoadState && !state.loadingState && Date.now() >= state.nextStateLoadAt) {
      await loadState();
    }
  } catch (error) {
    if (!state.loadingState) {
      $("metricStatus").textContent = "連線中斷";
      $("lastUpdated").textContent = error && error.message ? error.message : "暫時讀不到後台狀態，保留目前畫面";
    }
  } finally {
    state.loadingHeartbeat = false;
  }
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
  const reports = Array.isArray(data.reports) ? data.reports : state.reports;
  const reportCount = Number(data.report_count ?? reports.length);
  const passed = Number(data.passed_count ?? reports.filter((item) => item.passed).length);
  $("metricCount").textContent = reportCount;
  $("metricPassed").textContent = passed;
  $("metricPassScore").textContent = data.passing_score;
  $("metricExchange").textContent = data.exchange || data.meta?.exchange || "-";
  const scanCount = Number(data.scan_count || 0);
  const now = Date.now() / 1000;
  state.runtime = {
    running: Boolean(data.running),
    lastStarted: data.last_started || state.runtime.lastStarted,
    lastCompleted: data.last_completed || state.runtime.lastCompleted,
    serverTime: Number(data.server_time || now),
    receivedAt: now,
  };
  $("metricStatus").textContent = data.running ? `刷新中 #${scanCount + 1}` : `待命 #${scanCount}`;
  const stats = data.signal_statistics || data.meta?.signal_statistics || {};
  const gradeCounts = stats.grade_counts || {};
  $("metricCandidateAB").textContent = Number(gradeCounts.A || 0) + Number(gradeCounts.B || 0);
  $("metricCandidateX").textContent = Number(gradeCounts.X || stats.failed_signals || 0);
  renderStats(stats);
  if (data.provider_gap_count !== undefined) {
    $("metricDataGaps").textContent = Number(data.provider_gap_count || 0);
  } else {
    $("metricDataGaps").textContent = (data.providers || []).filter((provider) => {
      const text = `${provider.state || ""}`;
      return text.includes("未設定") || text.includes("讀不到") || text.includes("失敗") || text.includes("部分");
    }).length;
  }
  const reason = data.last_refresh_reason ? ` · ${data.last_refresh_reason}` : "";
  $("lastUpdated").textContent = data.last_completed ? `最後刷新：${new Date(data.last_completed).toLocaleString("zh-TW")} · 已掃 ${scanCount} 次${reason}` : "尚未完成首次刷新";
  updateCountdown(data.next_refresh_ts, data.server_time);
}

function renderStandoutAlerts(alerts = []) {
  const panel = $("obviousAlerts");
  const list = $("obviousAlertList");
  if (!panel || !list) return;
  const rows = Array.isArray(alerts) ? alerts : [];
  state.standoutAlerts = rows;
  panel.hidden = rows.length === 0;
  if (!rows.length) {
    list.innerHTML = "";
    return;
  }
  list.innerHTML = rows.map((item) => `
    <button class="standout-item" data-alert-symbol="${escapeHtml(item.symbol)}">
      <strong>${escapeHtml(item.symbol)}</strong>
      <span>${escapeHtml(item.direction_label || item.direction || "-")} · execution ${Number(item.execution_score || 0).toFixed(1)} · RR ${Number(item.rr || 0).toFixed(2)} · 距 entry ${Number(item.entry_distance_pct || 0).toFixed(2)}%</span>
      <b>查看</b>
    </button>
  `).join("");
}

function renderStats(stats = {}) {
  const node = $("statsSummary");
  if (!node) return;
  const accuracy = stats.accuracy || {};
  const gradeCounts = stats.grade_counts || {};
  const statusCounts = stats.status_counts || {};
  const cards = [
    ["總訊號", stats.total_signals ?? 0],
    ["A/B/C/D/X", `A ${gradeCounts.A || 0} / B ${gradeCounts.B || 0} / C ${gradeCounts.C || 0} / D ${gradeCounts.D || 0} / X ${gradeCounts.X || 0}`],
    ["狀態", Object.entries(statusCounts).map(([k, v]) => `${statusLabel(k)} ${v}`).join(" / ") || "-"],
    ["1 根K後", accuracy.after_1_candles?.rate === null || accuracy.after_1_candles?.rate === undefined ? "-" : `${accuracy.after_1_candles.rate}%`],
    ["3 根K後", accuracy.after_3_candles?.rate === null || accuracy.after_3_candles?.rate === undefined ? "-" : `${accuracy.after_3_candles.rate}%`],
    ["6/12 根K後", `${accuracy.after_6_candles?.rate ?? "-"}% / ${accuracy.after_12_candles?.rate ?? "-"}%`],
    ["平均 MFE", stats.average_mfe === null || stats.average_mfe === undefined ? "-" : `${Number(stats.average_mfe).toFixed(3)}%`],
    ["平均 MAE", stats.average_mae === null || stats.average_mae === undefined ? "-" : `${Number(stats.average_mae).toFixed(3)}%`],
    ["失效訊號", stats.failed_signals ?? 0],
  ];
  node.innerHTML = cards.map(([label, value]) => `<article class="stat-card"><span>${label}</span><strong>${value}</strong></article>`).join("");
}

function updateCountdown(nextTs, serverTs) {
  const localNow = Date.now() / 1000;
  state.countdown = {
    nextTs: Number(nextTs || 0),
    serverTs: Number(serverTs || localNow),
    receivedAt: localNow,
  };
  tickCountdown();
}

function tickCountdown() {
  const localNow = Date.now() / 1000;
  if (state.runtime.running) {
    const started = state.runtime.lastStarted ? Date.parse(state.runtime.lastStarted) / 1000 : localNow;
    const elapsed = Math.max(0, localNow - started);
    const minutes = Math.floor(elapsed / 60);
    const seconds = Math.floor(elapsed % 60);
    $("metricCountdown").textContent = `掃描中 ${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
    return;
  }
  const nextTs = Number(state.countdown.nextTs || 0);
  if (!nextTs) {
    $("metricCountdown").textContent = "--:--";
    return;
  }
  const elapsed = localNow - Number(state.countdown.receivedAt || localNow);
  const estimatedServerNow = Number(state.countdown.serverTs || localNow) + elapsed;
  const remain = Math.max(0, nextTs - estimatedServerNow);
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

function rowMatchesFilter(item, query, filter) {
  if (query && !String(item.symbol || "").toUpperCase().includes(query)) return false;
  if (filter === "passed" && !item.passed) return false;
  if (filter === "failed" && item.passed) return false;
  if (filter === "executable" && !item.should_execute) return false;
  if (["long", "short", "neutral"].includes(filter) && item.direction !== filter) return false;
  if (filter.startsWith("action_") && item.trade_action !== filter.replace("action_", "")) return false;
  return true;
}

function renderRows(options = {}) {
  const query = $("search").value.trim().toUpperCase();
  const filter = $("filter").value || "all";
  state.filteredReports = state.reports.filter((item) => rowMatchesFilter(item, query, filter));
  state.rows.resetScroll = Boolean(options.resetScroll);
  scheduleRowsRender();
}

function scheduleRowsRender() {
  if (state.rows.frame) cancelAnimationFrame(state.rows.frame);
  state.rows.frame = requestAnimationFrame(() => {
    state.rows.frame = null;
    renderVisibleRows();
  });
}

function renderVisibleRows() {
  const body = $("rows");
  const wrap = document.querySelector(".table-wrap");
  if (!body || !wrap) return;
  try {
    if (state.rows.resetScroll) {
      wrap.scrollTop = 0;
      state.rows.resetScroll = false;
    }
    const rows = state.filteredReports;
    if (!rows.length) {
      body.innerHTML = `<tr><td class="empty-row" colspan="${TABLE_COLUMN_COUNT}">沒有符合條件的資料</td></tr>`;
      return;
    }
    if (rows.length <= VIRTUALIZE_AFTER) {
      body.innerHTML = rows.map(rowHtml).join("");
      return;
    }

    const viewportHeight = Math.max(wrap.clientHeight || 0, 260);
    const start = Math.max(0, Math.floor(wrap.scrollTop / ROW_HEIGHT) - ROW_OVERSCAN);
    const visibleCount = Math.ceil(viewportHeight / ROW_HEIGHT) + ROW_OVERSCAN * 2;
    const end = Math.min(rows.length, start + visibleCount);
    const topHeight = start * ROW_HEIGHT;
    const bottomHeight = Math.max(0, (rows.length - end) * ROW_HEIGHT);
    const visibleRows = rows.slice(start, end).map(rowHtml).join("");

    body.innerHTML = `
      ${spacerRow(topHeight)}
      ${visibleRows}
      ${spacerRow(bottomHeight)}
    `;
  } catch (error) {
    console.error("render rows failed", error);
    body.innerHTML = `<tr><td class="empty-row" colspan="${TABLE_COLUMN_COUNT}">榜單渲染失敗，請重新整理或查看 console</td></tr>`;
  }
}

function spacerRow(height) {
  if (height <= 0) return "";
  return `<tr class="spacer-row" aria-hidden="true"><td colspan="${TABLE_COLUMN_COUNT}" style="height:${height}px"></td></tr>`;
}

function rowHtml(item) {
  const entry = item.entry_zone ? `${fmtPrice(item.entry_zone[0])}-${fmtPrice(item.entry_zone[1])}` : "-";
  const selectedClass = state.selected?.symbol === item.symbol ? " selected" : "";
  const direction = item.direction || item.selected_direction || "neutral";
  const volume = item.quote_volume_24h ?? item.volume ?? 0;
  return `
    <tr class="signal-row${selectedClass}" data-symbol="${escapeHtml(item.symbol)}">
      <td>${escapeHtml(item.rank)}</td>
      <td><strong>${escapeHtml(item.symbol)}</strong></td>
      <td><span class="badge ${sideClass(direction)}">${directionLabel(direction)}</span></td>
      <td><span class="badge ${actionClass(item.trade_action)}">${escapeHtml(item.trade_action_label || "-")}</span></td>
      <td><span class="badge ${executionClass(item.execution_label)}">${escapeHtml(item.execution_label || "-")}</span></td>
      <td class="plan-cell" title="${escapeHtml(item.execution_summary || "-")}">${escapeHtml(item.execution_summary || "-")}</td>
      <td><span class="badge ${item.passed ? "pass" : "fail"}">${item.passed ? "及格" : "未及格"}</span></td>
      <td><span class="badge ${gradeClass(item.candidate_grade)}">${escapeHtml(item.candidate_grade || item.grade || "-")}</span></td>
      <td>${escapeHtml(statusLabel(item.candidate_status || item.signal_state?.status || "-"))}</td>
      <td>${escapeHtml(trendLabel(item.score_trend || item.signal_state?.score_trend || "-"))}</td>
      <td>${Number(item.confirm_count || 0)} / ${Number(item.miss_count || 0)}</td>
      <td><span class="score">${Number(item.score || 0).toFixed(1)}</span></td>
      <td>${Number(item.data_completeness || 0).toFixed(0)}%</td>
      <td>${escapeHtml(item.grade)}</td>
      <td>${fmtPrice(item.price)}</td>
      <td>${Number(item.change_pct_24h || 0).toFixed(2)}%</td>
      <td>${fmtVolume(volume)}</td>
      <td>${escapeHtml(entry)}</td>
      <td>${fmtPrice(item.stop)}</td>
      <td>${fmtPrice(item.target)}</td>
      <td>${item.rr === null || item.rr === undefined ? "-" : Number(item.rr).toFixed(2)}</td>
    </tr>
  `;
}

function showDetail(item) {
  state.selected = item;
  scheduleRowsRender();
  $("detailTitle").textContent = `${item.symbol} · ${directionLabel(item.direction)} · ${Number(item.score).toFixed(1)} 分`;
  const model = item.score_model || {};
  $("detailMeta").textContent = `${item.trade_action_label || "未分類"} · ${item.trade_action_reason || ""} · 原始 score ${Number(item.score || 0).toFixed(1)} · 候選 ${item.candidate_grade || "-"} / ${statusLabel(item.candidate_status || "-")} · 共振 ${Number(model.bonus_score || item.bonus_score || 0).toFixed(1)} · 資料完整度 ${Number(item.data_completeness || 0).toFixed(0)}%`;
  renderList("detailReasons", item.selected_reasons || item.reasons || ["目前沒有足夠共振"]);
  renderList("detailWarnings", item.selected_warnings || item.warnings || ["無重大提醒"]);
  renderRiskPlan(item);
  renderDiagnostics(item);
  renderSignalState(item);
  renderExecutionPlan(item);
  renderTradePlan(item);
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

function renderDiagnostics(item) {
  const q = item.quant_diagnostics || {};
  const model = item.score_model || {};
  const skippedCore = Object.entries(model.skipped_core || {}).map(([name, reason]) => `${featureLabel(name)}：${reason}`);
  const bucketLines = Object.entries(model.bucket_scores || {}).map(([name, value]) => {
    const weight = Number((model.bucket_weights || {})[name] || 0);
    return `${bucketLabel(name)}：${Number(value).toFixed(1)} / 100，權重 ${weight.toFixed(0)}`;
  });
  const values = [
    `計分模型：${model.method || "核心分數 + 共振加分"}`,
    `原始 score：${Number(item.score || 0).toFixed(1)} / 100`,
    `核心分：${Number(model.core_score || 0).toFixed(1)} / 100，可用核心分母 ${Number(model.core_available_max || 0).toFixed(1)}。`,
    `共振加分：${Number(model.bonus_score || 0).toFixed(1)} / ${Number(model.bonus_available_max || 0).toFixed(1)}，不進核心分母。`,
    `HTF 背景：${Number(q.htf_context || 0).toFixed(1)} / 100`,
    `LTF 觸發：${Number(q.ltf_trigger || 0).toFixed(1)} / 100`,
    `入場品質：${Number(q.entry_quality || 0).toFixed(1)} / 100`,
    `風控品質：${Number(q.risk_reward_quality || 0).toFixed(1)} / 100`,
    `市場品質：${Number(q.market_api_quality || 0).toFixed(1)} / 100`,
    `額外共振：${Number(q.optional_confluence || 0).toFixed(1)} / 100`,
    `衍生品 API：${q.external_api_ok ? "已讀到" : "未讀到"}`,
    `ICT 核心：${q.core_ict_ok ? "完整" : "不完整"}`,
    `外部資料規則：${model.paid_api_rule || "未設定的付費 API 不扣分"}`,
  ];
  bucketLines.forEach((line) => values.push(`分層桶：${line}`));
  (model.score_adjustments || item.score_adjustments || []).forEach((line) => values.push(`校準限制：${line}`));
  if (q.derivative_warning) values.push(`衍生品風險：${q.derivative_warning}`);
  if (q.direction_conflict) values.push(`方向衝突：${q.direction_conflict}`);
  skippedCore.forEach((line) => values.push(`核心資料缺口：${line}`));
  renderList("detailDiagnostics", values);
}

function renderSignalState(item) {
  const s = item.signal_state || {};
  const values = [
    `候選等級：${item.candidate_grade || s.priority_level || "-"}，狀態：${statusLabel(s.status || item.candidate_status || "-")}`,
    `分數：目前 ${Number(s.current_score || item.score || 0).toFixed(1)}，上次 ${s.previous_score === null || s.previous_score === undefined ? "-" : Number(s.previous_score).toFixed(1)}，變化 ${Number(s.score_change || 0).toFixed(1)}`,
    `最高/最低：${s.highest_score ?? "-"} / ${s.lowest_score ?? "-"}`,
    `趨勢：${trendLabel(s.score_trend || item.score_trend || "-")}`,
    `confirm / miss：${Number(s.confirm_count || 0)} / ${Number(s.miss_count || 0)}`,
    `訊號年齡：${Number(s.signal_age_minutes || 0).toFixed(1)} 分鐘`,
    `人工動作：${s.stable_action?.label || item.trade_action_label || "-"}`,
    `原始模型：${item.raw_trade_action_label || "-"}`,
  ];
  if (s.stability_reason) values.push(`狀態原因：${s.stability_reason}`);
  (s.warning_reason || item.warning_reason || []).forEach((line) => values.push(`警告：${line}`));
  (s.invalid_reason || item.invalid_reason || []).forEach((line) => values.push(`失效：${line}`));
  renderFutureValidation(values, s.future_validation || item.future_validation || {});
  (s.behavior_analysis || []).forEach((line) => values.push(line));
  renderList("detailSignalState", values);
}

function renderFutureValidation(values, validation) {
  if (!validation || !validation.signal_price) {
    values.push("後續驗證：等待下一輪掃描累積。");
    return;
  }
  values.push(`訊號價格：${fmtPrice(validation.signal_price)}，MFE ${Number(validation.max_favorable_move || 0).toFixed(3)}%，MAE ${Number(validation.max_adverse_move || 0).toFixed(3)}%`);
  [1, 3, 6, 12].forEach((step) => {
    const key = step === 1 ? "after_1_candle" : `after_${step}_candles`;
    const result = validation[key];
    if (result) {
      values.push(`${step} 根 5m K 後：${result.direction_correct ? "方向正確" : "方向相反"}，價格 ${fmtPrice(result.price)}，變動 ${Number(result.move_pct || 0).toFixed(3)}%`);
    } else {
      values.push(`${step} 根 5m K 後：尚未完成驗證`);
    }
  });
  if (validation.failed_reason) values.push(`後續失敗原因：${validation.failed_reason}`);
}

function renderTradePlan(item) {
  const s = item.signal_state || {};
  renderList("detailTradePlan", s.trade_plan || ["尚未形成穩定交易計畫"]);
}

function renderExecutionPlan(item) {
  const values = [
    `判斷：${item.execution_label || "-"}，方式：${item.execution_mode || "-"}`,
    item.execution_summary || "尚未形成可執行計畫",
    ...(item.execution_steps || []),
  ];
  renderList("detailExecutionPlan", values);
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

function gradeClass(value) {
  const grade = String(value || "").toLowerCase();
  if (grade === "a") return "grade-a";
  if (grade === "b") return "grade-b";
  if (grade === "c") return "grade-c";
  if (grade === "d") return "grade-d";
  return "grade-x";
}

function statusLabel(value) {
  const labels = {
    new: "新出現",
    watching: "觀察中",
    active: "有效",
    strengthening: "增強",
    weakening: "轉弱",
    warning: "警告",
    invalid: "失效",
    expired: "過期",
    missed: "錯過",
  };
  return labels[value] || value || "-";
}

function trendLabel(value) {
  const labels = {
    new: "新訊號",
    strengthening: "增強",
    weakening: "轉弱",
    strong_jump: "快速轉強",
    sharp_drop: "快速轉弱",
    stable: "穩定",
  };
  return labels[value] || value || "-";
}

function bucketLabel(name) {
  const labels = {
    htf_context: "HTF 背景",
    ltf_confirmation: "LTF 觸發",
    entry_location: "入場位置",
    risk_plan: "風控計畫",
    market_filter: "行情品質",
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
  state.runtime.running = true;
  state.runtime.lastStarted = new Date().toISOString();
  tickCountdown();
  await api("/api/scan", { method: "POST", body: "{}" });
  setTimeout(loadHeartbeat, 600);
}

async function checkApis() {
  $("metricStatus").textContent = "測試 API";
  const data = await api("/api/check-apis", { method: "POST", body: "{}", timeoutMs: 30000 });
  renderProviders(data.providers || [], state.config);
  $("metricStatus").textContent = "待命";
}

$("saveSettings").addEventListener("click", saveSettings);
$("scanNow").addEventListener("click", scanNow);
$("checkApis").addEventListener("click", checkApis);
$("search").addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => renderRows({ resetScroll: true }), 120);
});
$("filter").addEventListener("change", () => renderRows({ resetScroll: true }));
$("rows").addEventListener("click", (event) => {
  const row = event.target.closest(".signal-row");
  if (!row) return;
  const item = state.filteredReports.find((report) => report.symbol === row.dataset.symbol);
  if (item) showDetail(item);
});
$("obviousAlertList").addEventListener("click", (event) => {
  const itemNode = event.target.closest("[data-alert-symbol]");
  if (!itemNode) return;
  const symbol = itemNode.getAttribute("data-alert-symbol");
  const item = state.reports.find((report) => report.symbol === symbol);
  if (item) showDetail(item);
});
document.querySelector(".table-wrap").addEventListener("scroll", () => {
  if (state.rows.scrollFrame) return;
  state.rows.scrollFrame = requestAnimationFrame(() => {
    state.rows.scrollFrame = null;
    renderVisibleRows();
  });
}, { passive: true });

loadState();
setInterval(tickCountdown, 1000);
startHeartbeatLoop();

function startHeartbeatLoop() {
  clearTimeout(heartbeatTimer);
  const run = async () => {
    await loadHeartbeat();
    heartbeatTimer = setTimeout(run, state.runtime.running ? HEARTBEAT_RUNNING_MS : HEARTBEAT_IDLE_MS);
  };
  heartbeatTimer = setTimeout(run, HEARTBEAT_RUNNING_MS);
}
