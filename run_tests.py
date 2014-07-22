import test_pendulum

for test in dir(test_pendulum):
    if test.startswith('test_'):
        getattr(test_pendulum, test)()
