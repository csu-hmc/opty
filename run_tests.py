import test_pendulum
import test_utils


for module in [test_pendulum, test_utils]:
    for test in dir(module):
        if test.startswith('test_'):
            getattr(module, test)()
