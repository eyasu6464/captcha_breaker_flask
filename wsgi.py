import captcha_breaker as myapp

# This is just a simple wrapper for gunicorn to find your app.
# If you want to change the algorithm file, simply change "captcha_breaker" above to the
# new file.

app = myapp.app
