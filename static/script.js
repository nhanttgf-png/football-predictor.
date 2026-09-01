const selDoiNha = document.getElementById("doi-nha");
const selDoiKhach = document.getElementById("doi-khach");
const btnPredict = document.getElementById("btn-predict");
const errorMsg = document.getElementById("error-msg");
const resultBox = document.getElementById("result");

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
      return;
    }

    document.getElementById("result-heading").textContent =
      `${data.doi_nha} (sân nhà) vs ${data.doi_khach} (sân khách)`;
    document.getElementById("label-home").textContent = data.doi_nha;
    document.getElementById("label-away").textContent = data.doi_khach;

    updateBar("bar-home", "pct-home", data.thang_nha);
    updateBar("bar-draw", "pct-draw", data.hoa);
    updateBar("bar-away", "pct-away", data.thang_khach);

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