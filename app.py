#----------------------------------------------------------------------------#
# Imports
#----------------------------------------------------------------------------#
import requests
import pickle
# from app import app
from flask import Blueprint, render_template, redirect, request, Response
from flask import Flask, render_template, request
# from flask.ext.sqlalchemy import SQLAlchemy
import logging
from logging import Formatter, FileHandler
from forms import *
import os

#----------------------------------------------------------------------------#
# App Config.
#----------------------------------------------------------------------------#


undecidedDict = {}
mythDict = {}

app = Flask(__name__)
app.config.from_object('config')
#db = SQLAlchemy(app)

# Automatically tear down SQLAlchemy.
'''
@app.teardown_request
def shutdown_session(exception=None):
    db_session.remove()
'''

# Login required decorator.
'''
def login_required(test):
    @wraps(test)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return test(*args, **kwargs)
        else:
            flash('You need to login first.')
            return redirect(url_for('login'))
    return wrap
'''
#----------------------------------------------------------------------------#
# Controllers.
#----------------------------------------------------------------------------#


@app.route('/demo')
def demo():
    with open('BaseCases.txt', 'r') as inf:
        mythDict = eval(inf.read())
    inf.close()
    return render_template('layouts/untitled.html', key_list=list(mythDict.keys()), val_list=list(mythDict.values()), len = len(mythDict))


@app.route('/submit', methods=['POST', 'GET'])
def submit_review():
    global mythDict
    post_content = request.form["content"]
    print(post_content)
    undecidedDict[post_content] = "undecided"
    new_dict = open("UndecidedCases.txt", 'w')
    new_dict.write(str(undecidedDict))
    new_dict.close()
    return render_template('layouts/untitled.html', key_list=list(mythDict.keys()), val_list=list(mythDict.values()), len = len(mythDict))


@app.route('/admin')
def admin():
    with open('UndecidedCases.txt', 'r') as inf2:
        undecidedDict = eval(inf2.read())
    inf2.close()
    return render_template('layouts/admin.html',  key_list=list(undecidedDict.keys()), val_list=list(undecidedDict.values()), len = len(undecidedDict))

@app.route('/submitExp', methods=['POST', 'GET'])
def submit_exp():
    explanation = request.form["exp"]
    num = request.form["bdh"]
    print(explanation)
    print(num)
    with open('BaseCases.txt', 'r') as inf:
        mythDict = eval(inf.read())
    inf.close()
    with open('UndecidedCases.txt', 'r') as inf2:
        undecidedDict = eval(inf2.read())
    inf2.close()
    unKey_list = list(undecidedDict.keys())
    print(len(unKey_list))
    mythDict[unKey_list[int(num)-1]] = explanation
    new_dict = open("BaseCases.txt", 'w')
    new_dict.write(str(mythDict))
    new_dict.close()
    undecidedDict.pop(unKey_list[int(num)-1])
    new_dict = open("UndecidedCases.txt", 'w')
    new_dict.write(str(undecidedDict))
    new_dict.close()
    return render_template('layouts/admin.html',  key_list=list(undecidedDict.keys()), val_list=list(undecidedDict.values()), len = len(undecidedDict))


@app.route('/')
def home():
    return render_template('wavefire.html')
    # return render_template('pages/placeholder.home.html')


@app.route('/about')
def about():
    return render_template('pages/placeholder.about.html')


@app.route('/login')
def login():
    form = LoginForm(request.form)
    return render_template('forms/login.html', form=form)


@app.route('/register')
def register():
    form = RegisterForm(request.form)
    return render_template('forms/register.html', form=form)


@app.route('/forgot')
def forgot():
    form = ForgotForm(request.form)
    return render_template('forms/forgot.html', form=form)

# Error handlers.


@app.errorhandler(500)
def internal_error(error):
    # db_session.rollback()
    return render_template('errors/500.html'), 500


@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404


if not app.debug:
    file_handler = FileHandler('error.log')
    file_handler.setFormatter(
        Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    )
    app.logger.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.info('errors')

#----------------------------------------------------------------------------#
# Launch.
#----------------------------------------------------------------------------#

# Default port:
if __name__ == '__main__':
    app.run()

# Or specify port manually:
'''
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
'''
