import os
import subprocess

epochs=(1,)
batch_sizes=(16,32,64)
n_workers=(50,75,100)
learning_rates=(0.001,0.000181,0.00001)
weight_decays=(0.0,0.01)
kl_loss_weights=(0.0,)
max_batches=(1000,2700,5000)


if __name__ == '__main__':
    # remove .env file
    for epoch in epochs:
        for batch_size in batch_sizes:
            for n_worker in n_workers:
                if n_worker > batch_size:
                    for learning_rate in learning_rates:
                        for weight_decay in weight_decays:
                            for kl_loss_weight in kl_loss_weights:
                                for max_batch in max_batches:
                                    os.remove(".env")
                                    with open(".env", "w+") as f:
                                        f.write(f"NAME=kulbach_{max_batch}_{n_worker}_{learning_rate}_{weight_decay}_{kl_loss_weight}\n")
                                        f.write("VERSION='0_0_1'\n")
                                        f.write("MODELS_ROOT='/home/shared/BASALT/models'\n")
                                        f.write("PORT=9898\n")
                                        f.write("PYTHONUNBUFFERED=1\n")
                                        f.write("EVALUATION_STAGE=testing\n")
                                        f.write("GIT_ACCESS_TOKEN=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\n")
                                        f.write("DATA_ROOT=data_wombat\n")
                                        f.write(f"EPOCHS={epoch}\n")
                                        f.write(f"BATCH_SIZE={batch_size}\n")
                                        f.write(f"N_WORKERS={n_worker}\n")
                                        f.write(f"LEARNING_RATE={learning_rate}\n")
                                        f.write(f"WEIGHT_DECAY={weight_decay}\n")
                                        f.write(f"KL_LOSS_WEIGHT={kl_loss_weight}\n")
                                        f.write(f"MAX_BATCHES={max_batch}\n")
                                        f.close()
                                        subprocess.call(['sh',
                                                         './run.sh'])  # Thanks @Jim Dennis for suggesting the []

