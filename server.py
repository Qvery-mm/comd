import tornado.ioloop, tornado.httpserver, tornado.web, tornado.websocket
import os, random, string, io, socket, json
import image_process
from neuron import *
#from neuron import predict
from PIL import Image
from time import sleep
from light import kostil
from blank import anketa
from time import sleep
from math import sqrt

prediction = ""
verdict = ""
fname = ""
ttt = 0


class MainHandler(tornado.web.RequestHandler):

    def get(self):
        self.render("index.html", title="My title")

    def post(self):
        global verdict, prediction, fname, ttt
        try:
            # get photo
            file1 = self.request.files['file'][0]
            original_fname = file1['filename']
            extension = os.path.splitext(original_fname)[1]
            fname = ''.join(random.choice(string.ascii_lowercase + string.digits) for x in range(6))
            final_filename = fname + extension
            output_file = open("uploads/" + final_filename, 'wb')
            output_file.write(file1['body'])
            fname = final_filename
            name = self.get_argument("name")
            age = self.get_argument("age")
            gender = self.get_argument("gender")
            smoking =  self.get_argument("smoking")
        except:
            self.redirect("/")
            return 0

        #self.write("file " + original_fname + " is uploaded ")
        #image_process.work("uploads/" + final_filename)
        #show("uploads/" + final_filename)
        #Image.open(io.BytesIO(file1['body'])).show()  #opening bite-like object
        try:
            answer = 0
            answer = predict("uploads/" + final_filename)
            ttt = min(98, max(1, round(100*answer, 1) - 5 + round(sqrt(float(age)), 1)))
            if answer < 0.5:
                ttt += 0.05
                answer = "здоров"
            else:
                answer = "болен"
                ttt -= 0.05
            verdict = answer
            prediction = anketa(gender, age, smoking)
            self.render("processing.html")
        except Exception:
            pass

CONNECTIONS = []

class WSHandler(tornado.websocket.WebSocketHandler):

    #print("verdict", verdict)
    # def get(self):
    #     self.render("HelloWorld.html")

    def open(self):
        global verdict, prediction, fname, CONNECTIONS, ttt
        if self not in CONNECTIONS:
            CONNECTIONS.append(self)
        # for i in range(10):
        #     self.write_message(json.dumps({"type": "processing", "stat": i}))
        #     sleep(1)





    def on_message(self, message):
        # Reverse Message and send it back
        print('message', message.lower().rstrip() )
        if message.lower().rstrip() == 'start':
            kostil(CONNECTIONS, "uploads/" + fname)
            self.write_message(json.dumps({"type": "finish", "verdict": verdict, "prediction": prediction, "cap": ttt}))

    def on_close(self):
        if self in CONNECTIONS:
            CONNECTIONS.remove(self)

        print('connection closed')

    def check_origin(self, origin):
        return True



settings = {"static_path": os.path.join(os.path.dirname(__file__), "static")}

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r'/ws', WSHandler),
    ], **settings)

if __name__ == "__main__":
    app = make_app()
    app.listen(8888, address='0.0.0.0')
    tornado.ioloop.IOLoop.current().start()