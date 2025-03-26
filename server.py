from tornado.httpserver import HTTPServer
from tornado.wsgi import WSGIContainer
from tornado.ioloop import IOLoop
from app import app  # 导入你的Flask应用

if __name__ == '__main__':
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(5000)  # 监听本地5000端口
    print("Server started on port 5000")
    IOLoop.current().start()
