const selGiaiDau = document.getElementById("giai-dau");

// ---- Tab Dự đoán / Bảng xếp hạng ----
const tabPredict = document.getElementById("tab-predict");
const tabLeaderboard = document.getElementById("tab-leaderboard");
const predictView = document.getElementById("predict-view");
const leaderboardView = document.getElementById("leaderboard-view");
const leaderboardError = document.getElementById("leaderboard-error");
const leaderboardNote = document.getElementById("leaderboard-note");

// ---- Tab Cầu thủ ----
const tabPlayers = document.getElementById("tab-players");
const playersView = document.getElementById("players-view");
const playersError = document.getElementById("players-error");
const playersNote = document.getElementById("players-note");
const playersTeamFilter = document.getElementById("players-team-filter");
const playersPositionFilter = document.getElementById("players-position-filter");
const playersSort = document.getElementById("players-sort");
const playersTbody = document.getElementById("players-tbody");

// ---- Tab Lịch sử dự đoán ----
const tabHistory = document.getElementById("tab-history");
const historyView = document.getElementById("history-view");
const historyError = document.getElementById("history-error");
const historyNote = document.getElementById("history-note");
const historySummary = document.getElementById("history-summary");
const historyTbody = document.getElementById("history-tbody");

// ---- Khối độ tin cậy / tỷ số / phong độ / so sánh (trong kết quả dự đoán) ----
const confidenceBlock = document.getElementById("confidence-block");
const confidenceBadge = document.getElementById("confidence-badge");
const confidenceBar = document.getElementById("confidence-bar");
const confidenceReasons = document.getElementById("confidence-reasons");
const scoreBlock = document.getElementById("score-block");
const scoreBest = document.getElementById("score-best");
const scoreList = document.getElementById("score-list");
const formBlock = document.getElementById("form-block");
const formTeamHome = document.getElementById("form-team-home");
const formTeamAway = document.getElementById("form-team-away");
const formDotsHome = document.getElementById("form-dots-home");
const formDotsAway = document.getElementById("form-dots-away");
const compareBlock = document.getElementById("compare-block");
const compareColHome = document.getElementById("compare-col-home");
const compareColAway = document.getElementById("compare-col-away");
const compareTbody = document.getElementById("compare-tbody");

// ---- Cầu thủ nổi bật (trong kết quả dự đoán) ----
const keyPlayersBlock = document.getElementById("key-players-block");
const keyPlayersTeamHome = document.getElementById("key-players-team-home");
const keyPlayersTeamAway = document.getElementById("key-players-team-away");
const keyPlayersListHome = document.getElementById("key-players-list-home");
const keyPlayersListAway = document.getElementById("key-players-list-away");

// ---- Sub-tab Mùa giải hiện tại / Tổng hợp nhiều mùa ----
const subtabSeason = document.getElementById("subtab-season");
const subtabAlltime = document.getElementById("subtab-alltime");
const seasonTableBox = document.getElementById("season-table-box");
const alltimeTableBox = document.getElementById("alltime-table-box");
const seasonTableTbody = document.getElementById("season-table-tbody");
const leaderboardSort = document.getElementById("leaderboard-sort");
const leaderboardTbody = document.getElementById("leaderboard-tbody");
const selDoiNha = document.getElementById("doi-nha");
const selDoiKhach = document.getElementById("doi-khach");
const btnPredict = document.getElementById("btn-predict");
const errorMsg = document.getElementById("error-msg");
const resultBox = document.getElementById("result");
const usageNote = document.getElementById("usage-note");

// ---- Thanh tài khoản ----
const accountGuest = document.getElementById("account-guest");
const accountUser = document.getElementById("account-user");
const accountEmail = document.getElementById("account-email");
const accountBadge = document.getElementById("account-badge");
const btnOpenLogin = document.getElementById("btn-open-login");
const btnOpenRegister = document.getElementById("btn-open-register");
const btnLogout = document.getElementById("btn-logout");

// ---- Modal đăng nhập / đăng ký ----
const authModal = document.getElementById("auth-modal");
const authModalTitle = document.getElementById("auth-modal-title");
const authForm = document.getElementById("auth-form");
const authEmail = document.getElementById("auth-email");
const authPassword = document.getElementById("auth-password");
const authError = document.getElementById("auth-error");
const btnAuthCancel = document.getElementById("btn-auth-cancel");
const btnAuthSubmit = document.getElementById("btn-auth-submit");
const btnAuthSwitch = document.getElementById("btn-auth-switch");
const authSwitchText = document.getElementById("auth-switch-text");

// ---- Modal nâng cấp Premium ----
const upgradeModal = document.getElementById("upgrade-modal");
const upgradeModalError = document.getElementById("upgrade-modal-error");
const btnUpgrade = document.getElementById("btn-upgrade");
const btnModalCancel = document.getElementById("btn-modal-cancel");
const upgradePrice = document.getElementById("upgrade-price");
const upgradeQrBox = document.getElementById("upgrade-qr-box");
const upgradeQrImg = document.getElementById("upgrade-qr-img");
const upgradeBankName = document.getElementById("upgrade-bank-name");
const upgradeAccountNo = document.getElementById("upgrade-account-no");
const upgradeAccountName = document.getElementById("upgrade-account-name");
const upgradeTransferContent = document.getElementById("upgrade-transfer-content");

// ---- Khối kết quả chi tiết ----
const explainList = document.getElementById("explain-list");
const premiumLocked = document.getElementById("premium-locked");
const premiumStats = document.getElementById("premium-stats");
const statsColHome = document.getElementById("stats-col-home");
const statsColAway = document.getElementById("stats-col-away");
const statsTbody = document.getElementById("stats-tbody");

let authMode = "login"; // "login" | "register"
let currentUser = null; // { email, premium } hoặc null nếu là khách

function showError(message) {
  errorMsg.textContent = message;
  errorMsg.hidden = false;
}

function clearError() {
  errorMsg.hidden = true;
  errorMsg.textContent = "";
}

function fillSelect(select, teams) {
  select.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Chọn đội...";
  select.appendChild(placeholder);

  for (const team of teams) {
    const opt = document.createElement("option");
    opt.value = team;
    opt.textContent = team;
    select.appendChild(opt);
  }
  select.disabled = false;
}

// ==================== TÀI KHOẢN (đăng nhập / đăng ký / đăng xuất) ====================

function renderAccount() {
  if (currentUser) {
    accountGuest.hidden = true;
    accountUser.hidden = false;
    accountEmail.textContent = currentUser.email;
    accountBadge.hidden = !currentUser.premium;
    tabHistory.hidden = false;
  } else {
    accountGuest.hidden = false;
    accountUser.hidden = true;
    tabHistory.hidden = true;
    // Nếu khách đăng xuất trong lúc đang xem tab lịch sử, quay về tab dự đoán.
    if (!historyView.hidden) switchToPredictView();
  }
}

async function loadMe() {
  try {
    const res = await fetch("/api/me");
    const data = await res.json();
    currentUser = data.logged_in ? { email: data.email, premium: data.premium } : null;
  } catch (err) {
    currentUser = null;
  }
  renderAccount();
}

function openAuthModal(mode) {
  authMode = mode;
  authError.hidden = true;
  authError.textContent = "";
  authForm.reset();

  if (mode === "register") {
    authModalTitle.textContent = "Đăng ký";
    btnAuthSubmit.textContent = "Đăng ký";
    authSwitchText.textContent = "Đã có tài khoản?";
    btnAuthSwitch.textContent = "Đăng nhập";
  } else {
    authModalTitle.textContent = "Đăng nhập";
    btnAuthSubmit.textContent = "Đăng nhập";
    authSwitchText.textContent = "Chưa có tài khoản?";
    btnAuthSwitch.textContent = "Đăng ký";
  }

  authModal.hidden = false;
  authEmail.focus();
}

function closeAuthModal() {
  authModal.hidden = true;
}

async function submitAuth(e) {
  e.preventDefault();
  authError.hidden = true;

  const email = authEmail.value.trim();
  const password = authPassword.value;
  const endpoint = authMode === "register" ? "/api/register" : "/api/login";

  btnAuthSubmit.disabled = true;
  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();

    if (!res.ok) {
      authError.textContent = data.error || "Có lỗi xảy ra, vui lòng thử lại.";
      authError.hidden = false;
      return;
    }

    currentUser = { email: data.email, premium: data.premium };
    renderAccount();
    closeAuthModal();
    refreshUsageNote();
  } catch (err) {
    authError.textContent = "Không kết nối được tới server.";
    authError.hidden = false;
  } finally {
    btnAuthSubmit.disabled = false;
  }
}

async function logout() {
  try {
    await fetch("/api/logout", { method: "POST" });
  } catch (err) {
    // vẫn xoá trạng thái ở FE dù request lỗi, tránh kẹt UI
  }
  currentUser = null;
  renderAccount();
  refreshUsageNote();
}

btnOpenLogin.addEventListener("click", () => openAuthModal("login"));
btnOpenRegister.addEventListener("click", () => openAuthModal("register"));
btnAuthCancel.addEventListener("click", closeAuthModal);
btnAuthSwitch.addEventListener("click", () => openAuthModal(authMode === "login" ? "register" : "login"));
authForm.addEventListener("submit", submitAuth);
btnLogout.addEventListener("click", logout);

// ==================== GIỚI HẠN LƯỢT DÙNG MIỄN PHÍ ====================

function renderUsage(usage) {
  if (!usage) {
    usageNote.textContent = "";
    return;
  }
  usageNote.classList.remove("usage-low", "usage-premium");

  if (usage.premium) {
    usageNote.textContent = "⭐ Tài khoản Premium — dự đoán không giới hạn.";
    usageNote.classList.add("usage-premium");
    return;
  }

  usageNote.textContent = `Còn ${usage.remaining}/${usage.limit} lượt dự đoán miễn phí hôm nay.`;
  if (usage.remaining <= 1) {
    usageNote.classList.add("usage-low");
  }
}

async function refreshUsageNote() {
  try {
    const res = await fetch("/api/usage");
    if (!res.ok) return;
    const usage = await res.json();
    renderUsage(usage);
  } catch (err) {
    // không hiển thị được thì thôi
  }
}

// ==================== NÂNG CẤP PREMIUM (Stripe Checkout) ====================

async function openUpgradeModal() {
  upgradeModalError.hidden = true;
  if (!currentUser) {
    closeUpgradeModal();
    openAuthModal("login");
    return;
  }

  upgradeQrBox.hidden = true;
  upgradeModal.hidden = false;

  try {
    const res = await fetch("/api/premium-qr");
    const data = await res.json();

    if (!res.ok) {
      upgradeModalError.textContent = data.error || "Không tải được thông tin thanh toán.";
      upgradeModalError.hidden = false;
      return;
    }

    upgradePrice.textContent = `${data.amount.toLocaleString("vi-VN")}đ / tháng`;
    upgradeQrImg.src = data.qr_url;
    upgradeBankName.textContent = data.bank_id;
    upgradeAccountNo.textContent = data.account_no;
    upgradeAccountName.textContent = data.account_name;
    upgradeTransferContent.textContent = data.transfer_content;
    upgradeQrBox.hidden = false;
  } catch (err) {
    upgradeModalError.textContent = "Không kết nối được tới server.";
    upgradeModalError.hidden = false;
  }
}

function closeUpgradeModal() {
  upgradeModal.hidden = true;
}

btnModalCancel.addEventListener("click", closeUpgradeModal);
document.addEventListener("click", (e) => {
  if (e.target && e.target.id === "btn-upgrade") {
    openUpgradeModal();
  }
});

// ==================== GIẢI THÍCH + THỐNG KÊ CHI TIẾT (kết quả dự đoán) ====================

const STATS_LABELS = {
  elo: "Điểm Elo",
  form_5_tran: "Phong độ 5 trận gần nhất",
  diem_moi_tran: "Điểm trung bình/trận",
  ngay_nghi: "Số ngày nghỉ",
  bat_bai_lien_tiep: "Chuỗi bất bại (trận)",
  ban_thang_tb: "Bàn thắng TB/trận",
  ban_thua_tb: "Bàn thua TB/trận",
};

function renderExplain(explain) {
  explainList.innerHTML = "";
  for (const line of explain || []) {
    const li = document.createElement("li");
    li.textContent = line;
    explainList.appendChild(li);
  }
}

function renderPremiumStats(stats, doiNha, doiKhach) {
  if (!stats) {
    premiumLocked.hidden = false;
    premiumStats.hidden = true;
    return;
  }

  premiumLocked.hidden = true;
  premiumStats.hidden = false;
  statsColHome.textContent = doiNha;
  statsColAway.textContent = doiKhach;
  statsTbody.innerHTML = "";

  for (const [key, label] of Object.entries(STATS_LABELS)) {
    const row = stats[key];
    if (!row) continue;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${label}</td><td>${row.home}</td><td>${row.away}</td>`;
    statsTbody.appendChild(tr);
  }

  const h2h = stats.doi_dau_gan_day;
  if (h2h) {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td>Đối đầu ${h2h.so_tran} trận gần nhất</td>` +
      `<td>${h2h.home_thang} thắng</td>` +
      `<td>${h2h.away_thang} thắng</td>`;
    statsTbody.appendChild(tr);
  }
}

function renderConfidence(doTinCay) {
  if (!doTinCay) {
    confidenceBlock.hidden = true;
    return;
  }
  confidenceBlock.hidden = false;
  confidenceBadge.textContent = `${doTinCay.muc} · ${doTinCay.diem}/100`;
  confidenceBadge.className = "confidence-badge confidence-" +
    (doTinCay.muc === "CAO" ? "high" : doTinCay.muc === "THẤP" ? "low" : "mid");
  confidenceBar.style.width = `${doTinCay.diem}%`;
  confidenceReasons.innerHTML = "";
  for (const line of doTinCay.ly_do || []) {
    const li = document.createElement("li");
    li.textContent = line;
    confidenceReasons.appendChild(li);
  }
}

function renderScoreBlock(tySoChinhXac) {
  if (!tySoChinhXac || !tySoChinhXac.top5 || !tySoChinhXac.top5.length) {
    scoreBlock.hidden = true;
    return;
  }
  scoreBlock.hidden = false;
  scoreBest.textContent = tySoChinhXac.du_doan_nhat || "—";
  scoreList.innerHTML = "";
  for (const item of tySoChinhXac.top5) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="score-item-score">${item.ty_so}</span><span class="score-item-pct">${item.xac_suat}%</span>`;
    scoreList.appendChild(li);
  }
}

const KET_QUA_ICON = { W: "🟢", D: "🟡", L: "🔴" };

function renderFormDots(container, matches) {
  container.innerHTML = "";
  if (!matches || !matches.length) {
    container.textContent = "Chưa có dữ liệu.";
    return;
  }
  for (const m of matches) {
    const span = document.createElement("span");
    span.className = "form-dot form-dot-" + m.ket_qua.toLowerCase();
    span.textContent = KET_QUA_ICON[m.ket_qua] || "•";
    span.title = `${m.san === "nha" ? "Sân nhà" : "Sân khách"} vs ${m.doi_thu}: ${m.ty_so}`;
    container.appendChild(span);
  }
}

function renderFormChart(bieuDo, doiNha, doiKhach) {
  if (!bieuDo) {
    formBlock.hidden = true;
    return;
  }
  formBlock.hidden = false;
  formTeamHome.textContent = doiNha;
  formTeamAway.textContent = doiKhach;
  renderFormDots(formDotsHome, bieuDo.nha);
  renderFormDots(formDotsAway, bieuDo.khach);
}

const COMPARE_LABELS = {
  elo: "Điểm Elo",
  phong_do_pct: "Phong độ (%)",
  ban_thang_tb: "Bàn thắng TB/trận",
  ban_thua_tb: "Bàn thua TB/trận",
  win_rate_pct: "Tỷ lệ thắng (%)",
};

function renderCompare(soSanh, doiNha, doiKhach) {
  if (!soSanh) {
    compareBlock.hidden = true;
    return;
  }
  compareBlock.hidden = false;
  compareColHome.textContent = doiNha;
  compareColAway.textContent = doiKhach;
  compareTbody.innerHTML = "";
  for (const [key, label] of Object.entries(COMPARE_LABELS)) {
    const row = soSanh[key];
    if (!row) continue;
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${label}</td><td>${row.home}</td><td>${row.away}</td>`;
    compareTbody.appendChild(tr);
  }
}

// ==================== LỊCH SỬ DỰ ĐOÁN ====================

const HISTORY_LABEL = { nha: "Nhà thắng", hoa: "Hòa", khach: "Khách thắng" };
const HISTORY_PROB_FIELD = { nha: "thang_nha", hoa: "hoa", khach: "thang_khach" };

function historyRowHtml(row) {
  const ngay = row.created_at ? new Date(row.created_at).toLocaleDateString("vi-VN") : "—";
  const probField = HISTORY_PROB_FIELD[row.du_doan];
  const duDoanText = `${HISTORY_LABEL[row.du_doan] || row.du_doan} (${row[probField] ?? ""}%)`;
  let ketQuaHtml = '<span class="history-pending">Đang chờ kết quả</span>';
  if (row.ket_qua_thuc_te) {
    const dung = row.dung;
    ketQuaHtml = `<span class="history-${dung ? "correct" : "wrong"}">${HISTORY_LABEL[row.ket_qua_thuc_te] || row.ket_qua_thuc_te} ${dung ? "✅" : "❌"}</span>`;
  }
  return `
    <tr>
      <td>${ngay}</td>
      <td>${row.doi_nha} vs ${row.doi_khach}</td>
      <td>${duDoanText}</td>
      <td>${row.ty_so_du_doan || "—"}</td>
      <td>${ketQuaHtml}</td>
    </tr>`;
}

async function loadHistory() {
  historyError.hidden = true;
  historyNote.textContent = "Đang tải lịch sử dự đoán...";
  historySummary.innerHTML = "";
  historyTbody.innerHTML = "";
  try {
    const res = await fetch("/api/history");
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Không tải được lịch sử dự đoán.");

    historyNote.textContent = "";
    const tk = data.thong_ke || {};
    if (tk.da_co_ket_qua > 0) {
      historySummary.innerHTML =
        `<span class="history-summary-item">Độ chính xác AI: <strong>${tk.ty_le_chinh_xac}%</strong> ` +
        `(${tk.dung}/${tk.da_co_ket_qua} dự đoán đã có kết quả)</span>`;
    } else {
      historySummary.innerHTML =
        `<span class="history-summary-item">Chưa có dự đoán nào được xác nhận kết quả — kết quả sẽ tự cập nhật khi mô hình được huấn luyện lại với dữ liệu mới.</span>`;
    }

    if (!data.rows || !data.rows.length) {
      historyNote.textContent = "Bạn chưa có dự đoán nào. Hãy thử dự đoán 1 trận ở tab \"Dự đoán trận đấu\"!";
      return;
    }
    historyTbody.innerHTML = data.rows.map(historyRowHtml).join("");
  } catch (err) {
    historyError.textContent = err.message || "Có lỗi xảy ra khi tải lịch sử.";
    historyError.hidden = false;
    historyNote.textContent = "";
  }
}

async function loadLeagues() {
  try {
    const res = await fetch("/api/leagues");
    if (!res.ok) throw new Error("Không tải được danh sách giải đấu");
    const data = await res.json();

    selGiaiDau.innerHTML = "";
    for (const league of data.leagues) {
      const opt = document.createElement("option");
      opt.value = league.key;
      opt.textContent = league.name;
      selGiaiDau.appendChild(opt);
    }
    selGiaiDau.value = data.default;
    selGiaiDau.disabled = false;

    selGiaiDau.addEventListener("change", () => {
      resultBox.hidden = true;
      loadTeams(selGiaiDau.value);
      loadModelInfo(selGiaiDau.value);
      if (!leaderboardView.hidden) loadCurrentLeaderboardView();
      if (!playersView.hidden) loadPlayers();
    });

    // Tải đội + thông tin model cho giải mặc định
    loadTeams(data.default);
    loadModelInfo(data.default);
  } catch (err) {
    // Nếu không tải được danh sách giải (vd bản backend cũ chưa có
    // /api/leagues), vẫn cho web chạy bình thường với Ngoại hạng Anh.
    selGiaiDau.hidden = true;
    loadTeams();
    loadModelInfo();
  }
}

// ==================== BẢNG XẾP HẠNG ====================

let leaderboardSortLoaded = false;
let leaderboardSubview = "season"; // "season" | "alltime"

function setActiveTab(activeTab) {
  for (const tab of [tabPredict, tabLeaderboard, tabPlayers, tabHistory]) {
    const isActive = tab === activeTab;
    tab.classList.toggle("view-tab-active", isActive);
    tab.setAttribute("aria-selected", isActive ? "true" : "false");
  }
}

function switchToPredictView() {
  setActiveTab(tabPredict);
  predictView.hidden = false;
  leaderboardView.hidden = true;
  playersView.hidden = true;
  historyView.hidden = true;
}

function switchToLeaderboardView() {
  setActiveTab(tabLeaderboard);
  leaderboardView.hidden = false;
  predictView.hidden = true;
  playersView.hidden = true;
  historyView.hidden = true;
  loadCurrentLeaderboardView();
}

function switchToPlayersView() {
  setActiveTab(tabPlayers);
  playersView.hidden = false;
  predictView.hidden = true;
  leaderboardView.hidden = true;
  historyView.hidden = true;
  loadPlayers();
}

function switchToHistoryView() {
  setActiveTab(tabHistory);
  historyView.hidden = false;
  predictView.hidden = true;
  leaderboardView.hidden = true;
  playersView.hidden = true;
  loadHistory();
}

function switchToSeasonSubview() {
  leaderboardSubview = "season";
  subtabSeason.classList.add("lb-subtab-active");
  subtabSeason.setAttribute("aria-selected", "true");
  subtabAlltime.classList.remove("lb-subtab-active");
  subtabAlltime.setAttribute("aria-selected", "false");
  seasonTableBox.hidden = false;
  alltimeTableBox.hidden = true;
  loadCurrentLeaderboardView();
}

function switchToAlltimeSubview() {
  leaderboardSubview = "alltime";
  subtabAlltime.classList.add("lb-subtab-active");
  subtabAlltime.setAttribute("aria-selected", "true");
  subtabSeason.classList.remove("lb-subtab-active");
  subtabSeason.setAttribute("aria-selected", "false");
  alltimeTableBox.hidden = false;
  seasonTableBox.hidden = true;
  loadCurrentLeaderboardView();
}

function loadCurrentLeaderboardView() {
  if (leaderboardSubview === "season") loadSeasonTable();
  else loadLeaderboard();
}

function showLeaderboardError(message) {
  leaderboardError.textContent = message;
  leaderboardError.hidden = false;
}

function clearLeaderboardError() {
  leaderboardError.hidden = true;
  leaderboardError.textContent = "";
}

async function loadSeasonTable() {
  clearLeaderboardError();
  leaderboardNote.textContent = "";
  seasonTableTbody.innerHTML = `<tr><td colspan="10">Đang tải...</td></tr>`;

  try {
    const league = selGiaiDau.value;
    const url = `/api/season-table?league=${encodeURIComponent(league)}`;
    const res = await fetch(url);
    const data = await res.json();

    if (!res.ok) {
      seasonTableTbody.innerHTML = "";
      showLeaderboardError(data.error || "Có lỗi xảy ra, vui lòng thử lại.");
      return;
    }

    leaderboardNote.textContent = data.season
      ? `Bảng xếp hạng chính thức, mùa giải ${data.season}.`
      : "Bảng xếp hạng chính thức của mùa giải mới nhất.";

    renderSeasonTable(data.rows);
  } catch (err) {
    seasonTableTbody.innerHTML = "";
    showLeaderboardError("Không kết nối được tới server. Kiểm tra lại backend đang chạy chưa.");
  }
}

function renderSeasonTable(rows) {
  seasonTableTbody.innerHTML = "";

  if (!rows || !rows.length) {
    seasonTableTbody.innerHTML = `<tr><td colspan="10">Chưa có dữ liệu.</td></tr>`;
    return;
  }

  for (const r of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td class="lb-rank">${r.hang}</td>` +
      `<td class="lb-col-team">${r.doi}</td>` +
      `<td>${r.so_tran}</td>` +
      `<td>${r.thang}</td>` +
      `<td>${r.hoa}</td>` +
      `<td>${r.thua}</td>` +
      `<td>${r.bt}</td>` +
      `<td>${r.bb}</td>` +
      `<td>${r.hs > 0 ? "+" + r.hs : r.hs}</td>` +
      `<td>${r.diem}</td>`;
    seasonTableTbody.appendChild(tr);
  }
}

async function loadLeaderboard() {
  clearLeaderboardError();
  leaderboardTbody.innerHTML = `<tr><td colspan="9">Đang tải...</td></tr>`;

  try {
    const league = selGiaiDau.value;
    const sort = leaderboardSort.value || "diem";
    const url = `/api/leaderboard?league=${encodeURIComponent(league)}&sort=${encodeURIComponent(sort)}`;
    const res = await fetch(url);
    const data = await res.json();

    if (!res.ok) {
      leaderboardTbody.innerHTML = "";
      showLeaderboardError(data.error || "Có lỗi xảy ra, vui lòng thử lại.");
      return;
    }

    // Chỉ điền dropdown sắp xếp 1 lần (danh sách thông số không đổi giữa các giải)
    if (!leaderboardSortLoaded && data.sort_options && data.sort_options.length) {
      leaderboardSort.innerHTML = "";
      for (const opt of data.sort_options) {
        const o = document.createElement("option");
        o.value = opt.key;
        o.textContent = opt.label;
        leaderboardSort.appendChild(o);
      }
      leaderboardSort.value = data.sort;
      leaderboardSortLoaded = true;
    }

    leaderboardNote.textContent =
      "Xếp hạng dựa trên toàn bộ dữ liệu mô hình đã học (nhiều mùa giải), " +
      "không phải bảng xếp hạng chính thức của 1 mùa cụ thể.";

    renderLeaderboard(data.rows);
  } catch (err) {
    leaderboardTbody.innerHTML = "";
    showLeaderboardError("Không kết nối được tới server. Kiểm tra lại backend đang chạy chưa.");
  }
}

function renderLeaderboard(rows) {
  leaderboardTbody.innerHTML = "";

  if (!rows || !rows.length) {
    leaderboardTbody.innerHTML = `<tr><td colspan="9">Chưa có dữ liệu.</td></tr>`;
    return;
  }

  for (const r of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td class="lb-rank">${r.hang}</td>` +
      `<td class="lb-col-team">${r.doi}</td>` +
      `<td>${r.so_tran}</td>` +
      `<td>${r.diem}</td>` +
      `<td>${r.diem_moi_tran}</td>` +
      `<td>${r.ban_thang}</td>` +
      `<td>${r.ban_thua}</td>` +
      `<td>${r.hieu_so > 0 ? "+" + r.hieu_so : r.hieu_so}</td>` +
      `<td>${r.elo}</td>`;
    leaderboardTbody.appendChild(tr);
  }
}

tabPredict.addEventListener("click", switchToPredictView);
tabLeaderboard.addEventListener("click", switchToLeaderboardView);
tabPlayers.addEventListener("click", switchToPlayersView);
tabHistory.addEventListener("click", switchToHistoryView);
subtabSeason.addEventListener("click", switchToSeasonSubview);
subtabAlltime.addEventListener("click", switchToAlltimeSubview);
leaderboardSort.addEventListener("change", loadLeaderboard);

// ==================== CẦU THỦ (bảng rating) ====================

const POSITION_LABELS = { FW: "Tiền đạo", MF: "Tiền vệ", DF: "Hậu vệ", GK: "Thủ môn" };
let playersSortLoaded = false;
let currentTeams = [];

function showPlayersError(message) {
  playersError.textContent = message;
  playersError.hidden = false;
}

function clearPlayersError() {
  playersError.hidden = true;
  playersError.textContent = "";
}

function fillTeamFilter(select, teams) {
  const current = select.value;
  select.innerHTML = "";
  const allOpt = document.createElement("option");
  allOpt.value = "";
  allOpt.textContent = "Tất cả";
  select.appendChild(allOpt);
  for (const team of teams) {
    const opt = document.createElement("option");
    opt.value = team;
    opt.textContent = team;
    select.appendChild(opt);
  }
  select.value = teams.includes(current) ? current : "";
}

async function loadPlayers() {
  clearPlayersError();
  playersTbody.innerHTML = `<tr><td colspan="8">Đang tải...</td></tr>`;

  try {
    const league = selGiaiDau.value;
    const sort = playersSort.value || "rating";
    const params = new URLSearchParams({ league, sort });
    if (playersTeamFilter.value) params.set("team", playersTeamFilter.value);
    if (playersPositionFilter.value) params.set("position", playersPositionFilter.value);

    const res = await fetch(`/api/players?${params.toString()}`);
    const data = await res.json();

    if (!res.ok) {
      playersTbody.innerHTML = "";
      showPlayersError(data.error || "Có lỗi xảy ra, vui lòng thử lại.");
      return;
    }

    if (!playersSortLoaded && data.sort_options && data.sort_options.length) {
      playersSort.innerHTML = "";
      for (const opt of data.sort_options) {
        const o = document.createElement("option");
        o.value = opt.key;
        o.textContent = opt.label;
        playersSort.appendChild(o);
      }
      playersSort.value = data.sort;
      playersSortLoaded = true;
    }

    playersNote.textContent = data.season
      ? `Rating tính từ số liệu mùa giải ${data.season} trên FBref, chỉ tính cầu thủ đã đá đủ số phút tối thiểu.`
      : "Rating cầu thủ được tính từ số liệu mùa giải gần nhất trên FBref.";

    renderPlayers(data.rows);
  } catch (err) {
    playersTbody.innerHTML = "";
    showPlayersError("Không kết nối được tới server. Kiểm tra lại backend đang chạy chưa.");
  }
}

function renderPlayers(rows) {
  playersTbody.innerHTML = "";

  if (!rows || !rows.length) {
    playersTbody.innerHTML = `<tr><td colspan="8">Chưa có dữ liệu.</td></tr>`;
    return;
  }

  for (const r of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td class="lb-rank">${r.hang}</td>` +
      `<td class="lb-col-team">${r.cau_thu}</td>` +
      `<td class="lb-col-team">${r.doi}</td>` +
      `<td>${POSITION_LABELS[r.vi_tri] ? POSITION_LABELS[r.vi_tri][0] : (r.vi_tri || "?")}</td>` +
      `<td>${r.so_phut ?? ""}</td>` +
      `<td>${r.ban_thang ?? 0}</td>` +
      `<td>${r.kien_tao ?? 0}</td>` +
      `<td class="player-rating">${r.rating != null ? r.rating.toFixed(1) : "-"}</td>`;
    playersTbody.appendChild(tr);
  }
}

playersTeamFilter.addEventListener("change", loadPlayers);
playersPositionFilter.addEventListener("change", loadPlayers);
playersSort.addEventListener("change", loadPlayers);

// ==================== CẦU THỦ NỔI BẬT (kết quả dự đoán) ====================

function renderKeyPlayers(highlight, doiNha, doiKhach) {
  const hasAny = highlight && ((highlight.nha && highlight.nha.length) || (highlight.khach && highlight.khach.length));
  if (!hasAny) {
    keyPlayersBlock.hidden = true;
    return;
  }
  keyPlayersBlock.hidden = false;
  keyPlayersTeamHome.textContent = doiNha;
  keyPlayersTeamAway.textContent = doiKhach;
  renderKeyPlayersList(keyPlayersListHome, highlight.nha);
  renderKeyPlayersList(keyPlayersListAway, highlight.khach);
}

function renderKeyPlayersList(listEl, players) {
  listEl.innerHTML = "";
  if (!players || !players.length) {
    listEl.innerHTML = `<li class="key-players-empty">Chưa có dữ liệu cầu thủ.</li>`;
    return;
  }
  for (const p of players) {
    const li = document.createElement("li");
    li.innerHTML =
      `<span class="key-players-name">${p.cau_thu}</span>` +
      `<span class="key-players-pos">${POSITION_LABELS[p.vi_tri] || p.vi_tri || ""}</span>` +
      `<span class="key-players-rating">${p.rating != null ? p.rating.toFixed(1) : "-"}</span>`;
    listEl.appendChild(li);
  }
}

async function loadTeams(league) {
  try {
    const url = league ? `/api/teams?league=${encodeURIComponent(league)}` : "/api/teams";
    selDoiNha.disabled = true;
    selDoiKhach.disabled = true;
    btnPredict.disabled = true;
    const res = await fetch(url);
    if (!res.ok) throw new Error("Không tải được danh sách đội");
    const data = await res.json();
    currentTeams = data.teams || [];
    fillSelect(selDoiNha, currentTeams);
    fillSelect(selDoiKhach, currentTeams);
    fillTeamFilter(playersTeamFilter, currentTeams);
    btnPredict.disabled = false;
  } catch (err) {
    showError("Không tải được danh sách đội. Kiểm tra lại server backend.");
  }
}

function updateBar(barId, pctId, value) {
  document.getElementById(barId).style.width = `${value}%`;
  document.getElementById(pctId).textContent = `${value}%`;
}

async function predict() {
  clearError();

  const doiNha = selDoiNha.value;
  const doiKhach = selDoiKhach.value;

  if (!doiNha || !doiKhach) {
    showError("Vui lòng chọn cả hai đội.");
    return;
  }
  if (doiNha === doiKhach) {
    showError("Hai đội phải khác nhau.");
    return;
  }

  btnPredict.disabled = true;
  btnPredict.textContent = "Đang tính...";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        doi_nha: doiNha,
        doi_khach: doiKhach,
        league: selGiaiDau.value || undefined,
      }),
    });
    const data = await res.json();

    if (!res.ok) {
      showError(data.error || "Có lỗi xảy ra, vui lòng thử lại.");
      if (data.usage) renderUsage(data.usage);
      if (data.limit_reached) openUpgradeModal();
      return;
    }

    document.getElementById("result-heading").textContent =
      `${data.doi_nha} (sân nhà) vs ${data.doi_khach} (sân khách)`;
    document.getElementById("label-home").textContent = data.doi_nha;
    document.getElementById("label-away").textContent = data.doi_khach;

    updateBar("bar-home", "pct-home", data.thang_nha);
    updateBar("bar-draw", "pct-draw", data.hoa);
    updateBar("bar-away", "pct-away", data.thang_khach);

    renderExplain(data.explain);
    renderConfidence(data.do_tin_cay);
    renderScoreBlock(data.ty_so_chinh_xac);
    renderFormChart(data.bieu_do_phong_do, data.doi_nha, data.doi_khach);
    renderCompare(data.so_sanh, data.doi_nha, data.doi_khach);
    renderKeyPlayers(data.cau_thu_noi_bat, data.doi_nha, data.doi_khach);
    renderPremiumStats(data.premium_stats, data.doi_nha, data.doi_khach);
    renderUsage(data.usage);

    resultBox.hidden = false;
  } catch (err) {
    showError("Không kết nối được tới server. Kiểm tra lại backend đang chạy chưa.");
  } finally {
    btnPredict.disabled = false;
    btnPredict.textContent = "Xem tỷ lệ";
  }
}

async function loadModelInfo(league) {
  try {
    const url = league ? `/api/model-info?league=${encodeURIComponent(league)}` : "/api/model-info";
    const res = await fetch(url);
    if (!res.ok) return;
    const data = await res.json();
    const note = document.getElementById("model-accuracy-note");
    if (data.accuracy != null) {
      note.textContent =
        `Đo trên ${data.so_tran_test} trận gần nhất chưa dùng để train: ` +
        `độ chính xác ${data.accuracy}%.`;
    }
  } catch (err) {
    // không hiển thị được thì thôi, không phải lỗi nghiêm trọng
  }
}

btnPredict.addEventListener("click", predict);
loadLeagues();
loadMe();
refreshUsageNote();