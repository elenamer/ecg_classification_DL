from models.resnet import ResNet
from evaluation.metrics import F1Metric
from datasets.ptbxldataset import PTBXLDataset
from datasets.arr10000dataset import Arr10000Dataset
from datasets.datagenerator import DataGenerator
from datasets.mitbihardataset import MITBIHARDataset
import tensorflow as tf
import os

classes = "rhythm"

num_classes = 5

db_name = "mitdb"
choice = "static"
eval_p = "specific"
patient_id = "203"

# experiments/mitdb/static/specificpatient/201/models/"
exp_path = "experiments"+os.sep+db_name+os.sep+choice+os.sep+eval_p+"patient"+os.sep+patient_id
if not os.path.exists(exp_path):
    os.makedirs(exp_path)
    os.mkdir(exp_path+os.sep+"models")
    os.mkdir(exp_path+os.sep+"results")

dat = MITBIHARDataset(db_name)

## Warning: for now balance = True and False are treated the same and saved to same files (i.e. overwritten)
#dat.generate_train_set(eval_p,choice,True)
#dat.generate_val_set(eval_p,choice,False)
#dat.generate_test_set(eval_p,choice,False)

train = dat.load_dataset(eval_p,choice, 'train', patient_id )
val = dat.load_dataset(eval_p,choice, 'val', patient_id )
test = dat.load_dataset(eval_p,choice, 'test', patient_id )

if len(train) != 2:
    print("generate crossval splits")
    # we need to do random n-crossval splits for val and test
if len(val) != 2:
    # we need to do random split for val
    print("this dataset does not exist")
if len(test) != 2:
    print("prob. error")
    ## shouldn't ever happen
    # options for now are: 1 defined splits for patient-specific, 2 totally random splits for intra-patient, 3 random val split for inter-patient



'''dataset=PTBXLDataset(classes)
dataset.examine_database()
print("DONE")
crossval_split_id = 0

# Datasets
partition = dataset.get_crossval_split(crossval_split_id) #Get from dataset class # IDs (dict)
labels = dataset.get_labels() #Get from dataset class # Labels (dict)

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)
'''

model = ResNet(num_outputs=num_classes, blocks=[2,2], filters=[32, 64], kernel_size=[15,15])


inputs = tf.keras.layers.Input((200,1,), dtype='float32')
m1 = tf.keras.Model(inputs=inputs, outputs=model.call(inputs))


m1.summary()

# tf.keras.utils.plot_model(m1,
#     to_file="ResNet-model-shapes.png",
#     show_shapes=True,
#     show_dtype=False,
#     show_layer_names=False,
#     rankdir="TB",
#     expand_nested=False,
#     dpi=300)

opt = tf.keras.optimizers.Adam(lr=0.0001)

m1.compile(optimizer=opt, 
                #tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                loss='categorical_crossentropy',
                metrics='acc')

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5 )
log_f1 = F1Metric(train=train,validation=val, path=exp_path+os.sep+"models")

m1.fit(x=train[0],y=train[1], validation_data = val, callbacks = [es, log_f1], epochs = 2)


'''
# initialize the weights of the model
input_shape, _ = tf.compat.v1.data.get_output_shapes(train_data)
inputs = build_input_tensor_from_shape(input_shape, dtype=input_dtype, ignore_batch_dim=True)
model(inputs)

checkpoint = CustomCheckpoint(
    filepath=str(args.job_dir / 'epoch_{epoch:02d}' / 'model.weights'),
    data=(validation_data, val['y']),
    score_fn=f1,
    save_best_only=False,
    verbose=1)


logger = tf.keras.callbacks.CSVLogger(str(args.job_dir / 'history.csv'))


model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)

'''