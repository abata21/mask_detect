import time
def a(num):
    while True:
        num = num + 0.1
        try:
            if type(num) == int:
                print(num)
            else:
                pass
        except Exception:
            print('not_int')

def main():
    a(1)

if __name__ == "__main__":
    main()
