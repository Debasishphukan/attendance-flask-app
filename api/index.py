from flask import Flask, render_template

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route("/")
def home():
    return "Hello from Flask on Vercel!"

# --- Vercel requires a 'handler' callable for Python apps ---
def handler(environ, start_response):
    return app(environ, start_response)
