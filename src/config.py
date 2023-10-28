
class ModelConfig:
    LINEAR_LAYERS_HIDDEN = [300, 150]
    DROPOUT = 0.4

class EvalConfig:
    THRESHOLD=0.5

class TaskConfig:
    PRIZE_THRESHOLD=300_000
    VALIDATION_DATASET_RATE=0.1

class TrainConfig:
    EPOCHS=1000
    LEARNING_RATE=3e-4
    BATCH_SIZE = 32
    


