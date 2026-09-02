"""
payments.py
-----------
Thanh toán Premium THẬT qua Stripe Checkout (gói subscription trả định kỳ).

CẤU HÌNH CẦN THIẾT (biến môi trường — KHÔNG hard-code key vào code):
  STRIPE_SECRET_KEY      Secret key tài khoản Stripe của bạn (sk_live_...
                          hoặc sk_test_... khi test).
  STRIPE_PRICE_ID         Price ID kiểu "recurring" của gói Premium — tạo
                          trong Stripe Dashboard > Product catalog > thêm
                          sản phẩm "Premium" > thêm giá theo tháng.
  STRIPE_WEBHOOK_SECRET   Secret ký webhook (whsec_...). Lấy khi bạn tạo
                          1 webhook endpoint trong Stripe Dashboard trỏ về
                          <domain-that-app>/api/stripe-webhook, lắng nghe
                          sự kiện "checkout.session.completed" và
                          "customer.subscription.updated/deleted".
  APP_BASE_URL            Domain public của app (Stripe cần URL tuyệt đối
                          cho trang thành công/huỷ), vd https://abc.com

Nếu chưa cấu hình các biến trên, endpoint tạo phiên thanh toán sẽ trả lỗi
rõ ràng thay vì crash server hoặc âm thầm không hoạt động.

QUAN TRỌNG: server CHỈ set is_premium=True sau khi webhook đã được XÁC MINH
CHỮ KÝ từ Stripe (stripe.Webhook.construct_event) — không bao giờ tin dữ
liệu do frontend tự gửi lên để mở khoá Premium, tránh bị giả mạo request.
"""

import os
import logging
from datetime import datetime

import stripe
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user

from extensions import db
from db_models import User

logger = logging.getLogger(__name__)

payments_bp = Blueprint("payments", __name__, url_prefix="/api")

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_ID = os.environ.get("STRIPE_PRICE_ID", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://localhost:5000")


def _stripe_configured() -> bool:
    return bool(stripe.api_key and STRIPE_PRICE_ID)


@payments_bp.route("/create-checkout-session", methods=["POST"])
@login_required
def create_checkout_session():
    """FE gọi khi user (đã đăng nhập) bấm 'Nâng cấp Premium'. Trả về URL
    trang thanh toán do chính Stripe host — app KHÔNG tự xử lý số thẻ."""
    if not _stripe_configured():
        return jsonify({
            "error": "Server chưa cấu hình Stripe (thiếu STRIPE_SECRET_KEY / "
                     "STRIPE_PRICE_ID). Xem hướng dẫn ở đầu file payments.py."
        }), 500

    if current_user.is_premium:
        return jsonify({"error": "Bạn đã là thành viên Premium rồi."}), 400

    try:
        checkout_session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            customer_email=current_user.email,
            # client_reference_id + metadata: cách webhook biết Session này
            # thuộc về User nào trong DB của mình.
            client_reference_id=str(current_user.id),
            metadata={"user_id": str(current_user.id)},
            success_url=f"{APP_BASE_URL}/?checkout=success",
            cancel_url=f"{APP_BASE_URL}/?checkout=cancel",
        )
    except Exception as e:
        logger.error("Lỗi tạo Stripe Checkout Session: %s", e, exc_info=True)
        return jsonify({"error": "Không tạo được phiên thanh toán, thử lại sau."}), 500

    return jsonify({"url": checkout_session.url})


@payments_bp.route("/billing-portal", methods=["POST"])
@login_required
def billing_portal():
    """Cho user tự quản lý (đổi thẻ / huỷ subscription) qua trang Stripe
    Customer Portal có sẵn — không cần tự xây UI quản lý thanh toán."""
    if not current_user.stripe_customer_id:
        return jsonify({"error": "Bạn chưa có gói Premium nào để quản lý."}), 400
    try:
        portal_session = stripe.billing_portal.Session.create(
            customer=current_user.stripe_customer_id,
            return_url=f"{APP_BASE_URL}/",
        )
    except Exception as e:
        logger.error("Lỗi tạo billing portal session: %s", e, exc_info=True)
        return jsonify({"error": "Không mở được trang quản lý thanh toán."}), 500
    return jsonify({"url": portal_session.url})


@payments_bp.route("/stripe-webhook", methods=["POST"])
def stripe_webhook():
    """Stripe gọi endpoint này khi có sự kiện thanh toán/subscription."""
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature", "")

    if not STRIPE_WEBHOOK_SECRET:
        logger.warning("STRIPE_WEBHOOK_SECRET chưa cấu hình, bỏ qua webhook.")
        return jsonify({"error": "Webhook chưa được cấu hình."}), 500

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe.error.SignatureVerificationError) as e:
        logger.warning("Webhook signature không hợp lệ: %s", e)
        return jsonify({"error": "invalid signature"}), 400

    event_type = event["type"]
    obj = event["data"]["object"]

    if event_type == "checkout.session.completed":
        user_id = obj.get("client_reference_id") or (obj.get("metadata") or {}).get("user_id")
        user = User.query.get(int(user_id)) if user_id else None
        if user is not None:
            user.is_premium = True
            user.stripe_customer_id = obj.get("customer")
            user.stripe_subscription_id = obj.get("subscription")
            user.premium_since = datetime.utcnow()
            db.session.commit()
            logger.info("User %s đã nâng cấp Premium qua Stripe.", user.email)
        else:
            logger.warning("Webhook checkout.session.completed không map được user_id=%s", user_id)

    elif event_type in ("customer.subscription.deleted", "customer.subscription.updated"):
        sub_status = obj.get("status")
        customer_id = obj.get("customer")
        user = User.query.filter_by(stripe_customer_id=customer_id).first()
        if user is not None:
            user.is_premium = sub_status in ("active", "trialing")
            db.session.commit()
            logger.info("Cập nhật trạng thái Premium user %s -> %s (status=%s)",
                        user.email, user.is_premium, sub_status)

    return jsonify({"received": True})
