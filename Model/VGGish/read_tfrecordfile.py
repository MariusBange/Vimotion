import tensorflow as tf

# filename = "/home/minhdanh/Documents/VGGish/Thao_nhap.tfrecord"
filename = "/Users/marius/OneDrive/Uni/Bachelor/Bachelorarbeit/Model/movie_to_final_output/Proj_5_VGGish/Thao_nhap.tfrecord"
# Define a reader and read the next record
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename)
print("Thao")
# Decode the record read by the reader
# features = tf.parse_single_example(serialized_example) #, features=feature)
