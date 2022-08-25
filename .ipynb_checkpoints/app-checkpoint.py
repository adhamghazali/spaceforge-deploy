from flask import Flask
app = Flask(__name__)
@app.route('/')
def index():
    return "Hello world!" 

if __name__ == '__main__':
    app.debug = True
    app.run(host="0.0.0.0",port=80) #host="0.0.0.0" will make the page accessable
                                              #by going to http://[ip]:5000/ on any computer in 
                                                                          #the network.


