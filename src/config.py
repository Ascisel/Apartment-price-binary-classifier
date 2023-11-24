
class ModelConfig:
    LINEAR_LAYERS_HIDDEN = [300, 150]
    DROPOUT = 0.4

class EvalConfig:
    THRESHOLD=0.5

class TaskConfig:
    PRIZE_THRESHOLD=300_000
    MODELS_DIR = 'models'

class TrainConfig:
    EPOCHS=100
    LEARNING_RATE=3e-4
    BATCH_SIZE = 32
    VALIDATION_DATASET_RATE=0.1
    


