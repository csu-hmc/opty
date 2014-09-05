from pendulum import Identifier

num_links = 1
sample_rate = 100.0
init_type = 'random'
sensor_noise = True
duration = 10.0

for i in range(20):
    identifier = Identifier(num_links, duration, sample_rate, init_type,
                            sensor_noise, False, False)
    identifier.identify()
