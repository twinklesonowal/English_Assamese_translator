from pyngrok import ngrok

ngrok.set_auth_token("3EGtdGNMadYMRNVDeHTeGM2VXCs_7uxbjiADN5DZVcXZ2TUJk")
url = ngrok.connect(5000)
print("Share this link with friends:", url)
input("Press Enter to stop...")