from flask import Blueprint, render_template, request, redirect, url_for, session
from app import db
from app.models.AuthModel import User

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/')
def root_redirect():
    return redirect(url_for('auth.login'))

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            session['user_id'] = user.id_user
            session['username'] = user.username
            session['role'] = user.role
            return render_template('redirect.html',
                                   message='Login berhasil',
                                   redirect_url=url_for('dashboard.index'))
        return render_template('redirect.html',
                               message='Login gagal: username atau password salah',
                               redirect_url=url_for('auth.login'))
    return render_template('views/auth/login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        nama_lengkap = request.form['nama_lengkap']

        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return render_template('redirect.html',
                                   message='Username sudah digunakan',
                                   redirect_url=url_for('auth.register'))

        user = User(username=username, password=password, nama_lengkap=nama_lengkap, role='admin')
        db.session.add(user)
        db.session.commit()
        return render_template('redirect.html',
                               message='Registrasi berhasil, silakan login',
                               redirect_url=url_for('auth.login'))
    return render_template('views/auth/register.html')

@auth_bp.route('/logout')
def logout():
    session.clear()
    return render_template('redirect.html',
                           message='Anda telah logout',
                           redirect_url=url_for('auth.login'))
