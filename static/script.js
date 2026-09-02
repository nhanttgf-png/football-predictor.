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
const btnModalConfirm = document.getElementById("btn-modal-confirm");

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
  } else {
    accountGuest.hidden = false;
    accountUser.hidden = true;
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

function openUpgradeModal() {
  upgradeModalError.hidden = true;
  if (!currentUser) {
    closeUpgradeModal();
    openAuthModal("login");
    return;
  }
  upgradeModal.hidden = false;
}

function closeUpgradeModal() {
  upgradeModal.hidden = true;
}

async function confirmUpgrade() {
  upgradeModalError.hidden = true;
  btnModalConfirm.disabled = true;
  btnModalConfirm.textContent = "Đang chuyển hướng...";

  try {
    const res = await fetch("/api/create-checkout-session", { method: "POST" });
    const data = await res.json();

    if (!res.ok) {
      upgradeModalError.textContent = data.error || "Không tạo được phiên thanh toán.";
      upgradeModalError.hidden = false;
      return;
    }

    window.location.href = data.url;
  } catch (err) {
    upgradeModalError.textContent = "Không kết nối được tới server.";
    upgradeModalError.hidden = false;
  } finally {
    btnModalConfirm.disabled = false;
    btnModalConfirm.textContent = "Thanh toán qua Stripe";
  }
}

btnModalCancel.addEventListener("click", closeUpgradeModal);
btnModalConfirm.addEventListener("click", confirmUpgrade);
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

async function loadTeams() {
  try {
    const res = await fetch("/api/teams");
    if (!res.ok) throw new Error("Không tải được danh sách đội");
    const data = await res.json();
    fillSelect(selDoiNha, data.teams);
    fillSelect(selDoiKhach, data.teams);
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
      body: JSON.stringify({ doi_nha: doiNha, doi_khach: doiKhach }),
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

async function loadModelInfo() {
  try {
    const res = await fetch("/api/model-info");
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
loadTeams();
loadModelInfo();
loadMe();
refreshUsageNote();