num_process = 5
env_indexes = [200, 300]
part = int((env_indexes[1] - env_indexes[0]) / num_process)
split_env_indexes = [[env_indexes[0] + i * part, env_indexes[0] + (i + 1) * part] for i in range(num_process)]
print(split_env_indexes)
