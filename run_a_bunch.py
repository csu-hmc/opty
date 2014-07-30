from pendulum import run_identification

num_links = 1
duration = 5.0
init_type = 'known'

for sample_rate in [50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0]:
    run_identification(num_links, duration, sample_rate, init_type, False, False)

num_links = 1
duration = 5.0
init_type = 'close'

for sample_rate in [50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0]:
    run_identification(num_links, duration, sample_rate, init_type, False, False)

num_links = 1
duration = 5.0
init_type = 'random'

for i in range(10):
    for sample_rate in [50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0]:
        run_identification(num_links, duration, sample_rate, init_type, False, False)

num_links = 1
sample_rate = 100.0
init_type = 'close'

for duration in [1.0, 5.0, 10.0, 40.0, 60.0, 90.0, 120.0, 180.0, 240.0]:
    run_identification(num_links, duration, sample_rate, init_type, False, False)

num_links = 2
sample_rate = 100.0
init_type = 'random'

for i in range(10):
    for duration in [60.0, 90.0, 120.0, 180.0, 240.0]:
        run_identification(num_links, duration, sample_rate, init_type, False, False)

num_links = 3
sample_rate = 100.0
init_type = 'random'

for i in range(10):
    for duration in [60.0, 90.0, 120.0, 180.0, 240.0]:
        run_identification(num_links, duration, sample_rate, init_type, False, False)
