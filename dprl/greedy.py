# -*- coding: utf-8 -*-
# Copyright (c) Percy.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Greedy Method """

class Greedy(object):
  pass


def test():
  epsilon_list_g = [epsilon_level[1]] * 60
  for kk in range(24):
    print('kk', kk)
    comp_list = []
    index_list = []
    select_index = 0
    for gi in range(60):
      if epsilon_list_g[gi] == epsilon_level[1]:
        # try
        temp_el = deepcopy(epsilon_list_g)
        temp_el[gi] = epsilon_level[0]
        temp_x, temp_y = process_train_data(x_train, y_train, temp_el)
        model.fit(temp_x, temp_y / 2)
        y_pred = model.predict(x_test)
        acc = metrics.accuracy_score(y_test, y_pred, normalize=True)
        comp_list.append(acc)
        print('comp_list', comp_list)
        index_list.append(gi)

    v = 0
    for walk in range(len(comp_list)):
      print('acc', comp_list[walk])
      if comp_list[walk] > v:
        v = comp_list[walk]
        select_index = walk

    print('select_index', select_index)
    print('index', index_list)

    # select_index = np.argmax(comp_list, axis=0)
    ii = index_list[select_index]
    epsilon_list_g[ii] = epsilon_level[0]
    print(epsilon_list_g)

  x_train_g, y_train_g = process_train_data(x_train,
                                            y_train,
                                            epsilon_list_g)

  model.fit(x_train_g, y_train_g / 2)
  y_pred = model.predict(x_test)
  acc = metrics.accuracy_score(y_test, y_pred, normalize=True)
  return acc
