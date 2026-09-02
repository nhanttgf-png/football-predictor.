"""
auth.py
-------
Đăng ký / đăng nhập / đăng xuất bằng email + mật khẩu, dùng Flask-Login để
quản lý session (cookie đã ký, giống cơ chế session có sẵn trong app.py).

Mật khẩu KHÔNG BAO GIỜ lưu dạng plain text — chỉ lưu hash một chiều
(werkzeug.security.generate_password_hash, mặc định dùng scrypt).
"""

import re

from flask import Blueprint, request, jsonify
from flask_login import login_user, logout_user, login_required, current_user

from extensions import db
from db_models import User

auth_bp = Blueprint("auth", __name__, url_prefix="/api")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LEN = 6


@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not EMAIL_RE.match(email):
        return jsonify({"error": "Email không hợp lệ."}), 400
    if len(password) < MIN_PASSWORD_LEN:
        return jsonify({"error": f"Mật khẩu cần ít nhất {MIN_PASSWORD_LEN} ký tự."}), 400
    if User.query.filter_by(email=email).first() is not None:
        return jsonify({"error": "Email này đã được đăng ký."}), 409

    user = User(email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    login_user(user, remember=True)
    return jsonify({"email": user.email, "premium": user.is_premium}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    user = User.query.filter_by(email=email).first()
    if user is None or not user.check_password(password):
        # Cố tình trả cùng 1 thông báo cho cả 2 trường hợp (email không tồn
        # tại / sai mật khẩu) để tránh lộ thông tin email nào đã đăng ký.
        return jsonify({"error": "Email hoặc mật khẩu không đúng."}), 401

    login_user(user, remember=True)
    return jsonify({"email": user.email, "premium": user.is_premium})


@auth_bp.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return jsonify({"message": "Đã đăng xuất."})


@auth_bp.route("/me")
def me():
    if not current_user.is_authenticated:
        return jsonify({"logged_in": False})
    return jsonify({
        "logged_in": True,
        "email": current_user.email,
        "premium": current_user.is_premium,
    })
