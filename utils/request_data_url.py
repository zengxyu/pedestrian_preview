import requests
from requests.auth import HTTPBasicAuth


def get_zip(url, filename, email, password):
    resp = requests.get(url, auth=HTTPBasicAuth(email, password))  # 传输账号密码，通过basic认证，如果不需要basic认证可以把auth这一部分删除
    with open(filename, "wb") as code:
        code.write(resp.content)
    print(resp.status_code)


if __name__ == '__main__':
    url = "https://pan.dm-ai.com/s/MDxQF9jiBPmT8y5"
    get_zip(url=url, filename="data", email="", password="12345678")
