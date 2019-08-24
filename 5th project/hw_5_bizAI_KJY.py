# -*- coding: utf-8 -*-

# =============================================================================
# # BIZ&AI lab 6기 과제_5
# * 리뷰글 긍부정 예측 - ANN 모델 *

import tensorflow as tf
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tqdm import tqdm
import os, pickle, time, sys


os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

###############################################################################
#     data preprocessing
###############################################################################
def load_data_fn(data_file_path, data_file_name, w2v_model, w2v_size, max_word_num):
    
    okt = Okt()
    os.chdir(data_file_path)
    raw_x = pd.read_csv(data_file_name[0], encoding='cp949')['review'].values
    raw_y = pd.read_csv(data_file_name[0], encoding='cp949')['label'].values
    
    pos_list = ['Adjective', 'Adverb', 'Determiner', 'Noun', 'Number', 'Verb']
    review_data = [okt.pos(xx, norm=True, stem=True) for xx in tqdm(raw_x)]
    second_data = []
    zero = np.zeros(w2v_size, dtype=np.float32)
    
    for idx in range(len(review_data)):
        second_data.append([ i for i, y in review_data[idx] if y in pos_list ])
    
    for j in range(len(second_data)):        
        for tt, voca in enumerate(second_data[j]):
            try:
                second_data[j][tt] = w2v_model.wv[voca]
            except KeyError:
                second_data[j][tt] = zero
    
    for t in range(len(second_data)):
        if len(second_data[t]) > max_word_num:
            del second_data[t][80:]
        while 1:
            if len(second_data[t]) < max_word_num:
                second_data[t].insert(0, zero)
            else:
                break
    
    load_x = second_data
    load_y = []

    for i in range(len(raw_y)):
        try:
            load_y.append(np.eye(2, dtype = int)[raw_y[i]])
        except IndexError:
            pass
    
    raw_x, load_x, load_y = np.array(raw_x), np.array(load_x), np.array(load_y)
    
    with open(data_file_name[1], 'wb') as f:
        pickle.dump([raw_x, load_x, load_y], f)

    return raw_x, load_x, load_y
    
###############################################################################
#     create model
###############################################################################
def create_ann_model(tf_model_important_var_name, max_word_num, w2v_size):
    
    tf.reset_default_graph()
    
    x_data = tf.placeholder(tf.float32, [None, max_word_num, w2v_size], name='x_data')
    re_x_data = tf.reshape(x_data, [-1, max_word_num*w2v_size])
    y_data = tf.placeholder(tf.float32, [None, 2], name='y_data')

    he_init = tf.contrib.layers.variance_scaling_initializer()
    z_1 = tf.layers.dense(re_x_data, 512, activation=tf.nn.relu, kernel_initializer=he_init)
    d_1 = tf.nn.dropout(z_1, 0.1, name='d_1')

    w_2 = tf.Variable(tf.random_normal([512, 256], stddev=0.01), name='w_2')
    b_2 = tf.Variable(tf.zeros(shape=(256)), name='b_2')
    bn_2 = tf.layers.batch_normalization(tf.matmul(d_1, w_2)+b_2, name='bn_2')
    z_2 = tf.nn.relu(bn_2, name='z_2')
    d_2 = tf.nn.dropout(z_2, 0.1, name='d_1')
    
    w_3 = tf.Variable(tf.random_normal([256, 2], stddev=0.01), name='w_3') 
    u_3 = tf.matmul(d_2, w_3, name='u_3') 
 

    learning_rate = 0.001
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=u_3, labels=y_data), name='loss')
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    
    pred_y = tf.argmax(tf.nn.softmax(u_3), 1, name='pred_y') 
    print('pred_y_shape:', pred_y.get_shape)
    pred = tf.equal(pred_y, tf.argmax(y_data, 1), name='pred') 
    print('pred_shape:', pred.get_shape)
    acc = tf.reduce_mean(tf.cast(pred, tf.float32), name='acc') 
    
    for op in [x_data, y_data, loss, acc, pred_y]:
        tf.add_to_collection(tf_model_important_var_name, op)

    return x_data, y_data, loss, acc, pred_y, train 
    

###############################################################################
#     early_stopping_and_save_model
###############################################################################
def early_stopping_and_save_model(sess, saver, tf_model_path, save_model_name, input_vali_loss, early_stopping_patience, early_stopping_val_loss_list):
    if len(early_stopping_val_loss_list) != early_stopping_patience:
        early_stopping_val_loss_list = [99.99 for _ in range(early_stopping_patience)]
    
    early_stopping_val_loss_list.append(input_vali_loss)
    if input_vali_loss < min(early_stopping_val_loss_list[:-1]):
        os.chdir(tf_model_path)
        saver.save(sess, './{0}/{0}.ckpt'.format(save_model_name))
        early_stopping_val_loss_list.pop(0)
        
        return True, early_stopping_val_loss_list
    
    elif early_stopping_val_loss_list.pop(0) < min(early_stopping_val_loss_list):
        return False, early_stopping_val_loss_list
    
    else:
        return True, early_stopping_val_loss_list
    
###############################################################################
#     model train
###############################################################################
def model_train(x_data, y_data, loss, acc, pred_y, train, x_train, y_train, x_vali, y_vali, batch_size, epoch_num, tf_model_path, tf_model_name, early_stopping_patience):
    batch_index_list = list(range(0, x_train.shape[0], batch_size))
    vali_batch_index_list = list(range(0, x_vali.shape[0], batch_size))
    
    train_loss_list, vali_loss_list = [], []
    train_acc_list, vali_acc_list = [], []
    
    start_time = time.time()
    saver = tf.train.Saver()
    early_stopping_val_loss_list = []
    print('\n%s\n%s - training....'%('-'*100, tf_model_name))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
                
        for epoch in range(epoch_num):
            total_loss, total_acc, vali_total_loss, vali_total_acc = 0, 0, 0, 0
            processing_bar_var = [0, 0]
            
            train_random_seed = int(np.random.random()*10**4)
            for x in [x_train, y_train]:
                np.random.seed(train_random_seed)
                np.random.shuffle(x)
                
            for i in batch_index_list:
                batch_x, batch_y = x_train[i:i+batch_size], y_train[i:i+batch_size]
                
                processing_bar_var[0] += len(batch_x)
                processing_bar_print = int(processing_bar_var[0]*100/len(x_train))-processing_bar_var[1]
                if processing_bar_print != 0:
                    sys.stdout.write('-'*processing_bar_print)
                    sys.stdout.flush()
    
                processing_bar_var[1] += (int(processing_bar_var[0]*100/len(x_train))-processing_bar_var[1])
                
                _, loss_val, acc_val = sess.run([train, loss, acc], feed_dict={x_data: batch_x, y_data: batch_y})
                total_loss += loss_val
                total_acc += acc_val
                
            train_loss_list.append(total_loss/len(batch_index_list))
            train_acc_list.append(total_acc/len(batch_index_list))
            
            sys.stdout.write('\n#%4d/%d%s' % (epoch + 1, epoch_num, '  |  '))
            sys.stdout.write('Train_loss={:.4f} / Train_acc={:.4f}{}'.format(train_loss_list[-1], train_acc_list[-1], '  |  '))
            sys.stdout.flush()
                    
            for i in vali_batch_index_list:
                vali_batch_x, vali_batch_y = x_vali[i:i+batch_size], y_vali[i:i+batch_size]
                
                vali_loss_val, vali_acc_val = sess.run([loss, acc], feed_dict={x_data: vali_batch_x, y_data: vali_batch_y})
                vali_total_loss += vali_loss_val
                vali_total_acc += vali_acc_val
                    
            vali_loss_list.append(vali_total_loss/len(vali_batch_index_list))
            vali_acc_list.append(vali_total_acc/len(vali_batch_index_list))
            
            tmp_running_time = time.time() - start_time
            sys.stdout.write('Vali_loss={:.4f} / Vali_acc={:.4f}{}'.format(vali_loss_list[-1], vali_acc_list[-1], '  |  '))
            sys.stdout.write('%dm %5.2fs\n'%(tmp_running_time//60, tmp_running_time%60))
            sys.stdout.flush()
            
            bool_continue, early_stopping_val_loss_list = early_stopping_and_save_model(sess, saver, tf_model_path, tf_model_name, vali_loss_list[-1], early_stopping_patience, early_stopping_val_loss_list)
            if not bool_continue:
                print('{0}\nstop epoch : {1}\n{0}'.format('-'*100, epoch-early_stopping_patience+1))
                break
            
    
    running_time = time.time() - start_time
    print('%s\ntraining time : %d m  %5.2f s\n%s'%('*'*100, running_time//60, running_time%60, '*'*100))
    
    
    os.chdir(r'{}\{}'.format(tf_model_path, tf_model_name))
    epoch_list = [i for i in range(1, epoch+2)]
    graph_loss_list = [train_loss_list, vali_loss_list, 'r', 'b', 'loss', 'upper right', '{}_loss.png'.format(tf_model_name)]
    graph_acc_list = [train_acc_list, vali_acc_list, 'r--', 'b--', 'acc', 'lower right', '{}_acc.png'.format(tf_model_name)]
    for train_l_a_list, vali_l_a_list, trian_color, vali_color, loss_acc, legend_loc, save_png_name in [graph_loss_list, graph_acc_list]:
        plt.plot(epoch_list, train_l_a_list, trian_color, label='train_'+loss_acc)
        plt.plot(epoch_list, vali_l_a_list, vali_color, label='validation_'+loss_acc)
        plt.xlabel('epoch')
        plt.ylabel(loss_acc)
        plt.legend(loc=legend_loc)
        plt.title(tf_model_name)
        plt.savefig(save_png_name)
        plt.show()
     
    
###############################################################################
#     model test
###############################################################################
def model_test(x_test, y_test, tf_model_path, tf_model_name, tf_model_important_var_name):
    tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        os.chdir(tf_model_path)
        saver = tf.train.import_meta_graph('./{0}/{0}.ckpt.meta'.format(tf_model_name))
        saver.restore(sess, './{0}/{0}.ckpt'.format(tf_model_name))
        x_data, y_data, loss, acc, pred_y = tf.get_collection(tf_model_important_var_name)
        
        test_loss, test_acc, test_y_pred, tmp_y_true = sess.run([loss, acc, pred_y, y_data], feed_dict={x_data: x_test, y_data: y_test})
            
    test_y_true = np.argmax(tmp_y_true, axis=1)
    print(classification_report(test_y_true, test_y_pred, target_names=['Positive', 'Negative']))
    print(pd.crosstab(pd.Series(test_y_true), pd.Series(test_y_pred), rownames=['True'], colnames=['Predicted'], margins=True))
    print('\nTest_loss = {:.4f} / Test_acc = {:.4f}\n{:s}'.format(test_loss, test_acc, '='*100))
        
###############################################################################
#     predict label
###############################################################################
def predict_label(tf_model_path, tf_model_name, tf_model_important_var_name, raw_hw_test, x_hw_test, pred_hw_test_file_path, pred_hw_test_file_name):
    tf.reset_default_graph()
    os.chdir(tf_model_path)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.import_meta_graph('./{0}/{0}.ckpt.meta'.format(tf_model_name))
        saver.restore(sess, './{0}/{0}.ckpt'.format(tf_model_name))
        x_data, y_data, loss, acc, pred_y = tf.get_collection(tf_model_important_var_name)
        
        model_y_pred = sess.run(pred_y, feed_dict={x_data: x_hw_test})
        
    os.chdir(pred_hw_test_file_path)
    pd.DataFrame({'label':model_y_pred, 'review':raw_hw_test}).to_csv(pred_hw_test_file_name, index=False, encoding = 'cp949')

        

if __name__ == "__main__":
# =============================================================================
    data_file_path = r'C:\Users\ggy01\OneDrive\바탕 화면\BIZ LAB\project4'
    train_file_name = ['hw_4_train_data_KJY.csv', 'hw_5_train_data_pickle_KJY.pkl']
    validation_file_name = ['hw_4_validation_data_KJY.csv', 'hw_5_vali_data_pickle_KJY.pkl']
    test_file_name = ['hw_4_test_data_KJY.csv','hw_5_test_data_pickle_KJY.pkl']

    hw_test_file_path = r'C:\Users\ggy01\OneDrive\바탕 화면\BIZ LAB\project5'
    hw_test_file_name = ['hw_5_model_test_data_no_label.csv', 'hw_5_model_test_data_no_label_pickle_KJY.pkl']
    
    w2v_file_path = r'C:\Users\ggy01\OneDrive\바탕 화면\BIZ LAB\project4'
    os.chdir(w2v_file_path)
    w2v_model = Word2Vec.load('hw_4_word2vec_KJY.model')
    
    tf_model_path = r'C:\Users\ggy01\OneDrive\바탕 화면\BIZ LAB\tf_model'
    tf_model_name = 'hw_5_ANN_model_KJY'
    
    tf_model_important_var_name = 'important_vars_ops'
    
    pred_hw_test_file_path = r'C:\Users\ggy01\OneDrive\바탕 화면\BIZ LAB\project5'
    pred_hw_test_file_name = 'hw_5_ann_predict_label_KJY.csv'

# =============================================================================  

    w2v_size = 256
    max_word_num = 80
    
    early_stopping_patience = 10
    epoch_num = 100
    batch_size = 512
# =============================================================================   

    _, x_train, y_train = load_data_fn(data_file_path, train_file_name, w2v_model, w2v_size, max_word_num)
    _, x_vali, y_vali = load_data_fn(data_file_path, validation_file_name, w2v_model, w2v_size, max_word_num)
    _, x_test, y_test = load_data_fn(data_file_path, test_file_name, w2v_model, w2v_size, max_word_num)
    raw_hw_test, x_hw_test, _ = load_data_fn(hw_test_file_path, hw_test_file_name, w2v_model, w2v_size, max_word_num)
    
    print('train_shape : {} / {}'.format(x_train.shape, y_train.shape))
    print('validation_shape : {} / {}'.format(x_vali.shape, y_vali.shape))
    print('test_shape : {} / {}'.format(x_test.shape, y_test.shape))
    print('\nmodel_test_shape : {}'.format(x_hw_test.shape))
    
    x_data, y_data, loss, acc, pred_y, train = create_ann_model(tf_model_important_var_name, max_word_num, w2v_size)
    
    model_train(x_data, y_data, loss, acc, pred_y, train, x_train, y_train, x_vali, y_vali, batch_size, epoch_num, tf_model_path, tf_model_name, early_stopping_patience)
    model_test(x_test, y_test, tf_model_path, tf_model_name, tf_model_important_var_name)
    predict_label(tf_model_path, tf_model_name, tf_model_important_var_name, raw_hw_test, x_hw_test, pred_hw_test_file_path, pred_hw_test_file_name)
