import numpy as np


class Step_Scheduler():
    def __init__(self, num_sample, warm_up, step_lr, **kwargs):
        warm_up_num = warm_up * num_sample
        self.eval_interval = lambda epoch: 1 if (epoch+1) > step_lr[-1] else 5
        self.lr_lambda = lambda num: num / warm_up_num \
                                     if num < warm_up_num else \
                                     0.1 ** np.sum(np.array(step_lr) <= num // num_sample)

    def get_lambda(self):
        return self.eval_interval, self.lr_lambda


class Cosine_Scheduler():
    def __init__(self, num_sample, max_epoch, warm_up, **kwargs):
        warm_up_num = warm_up * num_sample
        max_num = max_epoch * num_sample
        self.eval_interval = lambda epoch: 1 if (epoch+1) > max_epoch - 10 else 5
        self.lr_lambda = lambda num: num / warm_up_num \
                                     if num < warm_up_num else \
                                     0.5 * (np.cos((num - warm_up_num) / (max_num - warm_up_num) * np.pi) + 1)

    def get_lambda(self):
        return self.eval_interval, self.lr_lambda
    

class CosineWarmRestarts_Scheduler():
    def __init__(self, num_sample, T_0, T_mult, eta_min, max_epoch, warm_up, **kwargs):
        warm_up_num = warm_up * num_sample
        self.eta_min = eta_min

        # Convert epochs to iteration units
        T_0 = T_0 * num_sample
        self.T_mult = T_mult

        def get_T_cur(num):
            """Compute current position inside restart cycle (iteration-wise)."""
            num -= warm_up_num

            if T_mult == 1:
                return num % T_0, T_0

            T_i = T_0
            while num >= T_i:
                num -= T_i
                T_i *= T_mult
            return num, T_i

        self.eval_interval = lambda epoch: 1 if (epoch+1) > max_epoch - 10 else 5

        def lr_lambda(num):
            if num < warm_up_num:
                return num / warm_up_num

            T_cur, T_i = get_T_cur(num)

            return eta_min + (1 - eta_min) * \
                   0.5 * (1 + np.cos(np.pi * T_cur / T_i))

        self.lr_lambda = lr_lambda

    def get_lambda(self):
        return self.eval_interval, self.lr_lambda



class CosineWarmRestartsDecay_Scheduler():
    def __init__(self,
                 num_sample,
                 T_0,
                 T_mult,
                 gamma,          # decay factor for max LR after each restart
                 eta_min,
                 max_epoch,
                 warm_up,
                 **kwargs):

        warm_up_num = warm_up * num_sample
        T_0 = T_0 * num_sample
        self.T_mult = T_mult
        self.gamma = gamma
        self.eta_min = eta_min

        def get_cycle_info(num):
            """
            Returns:
                T_cur  -> position inside current cycle
                T_i    -> length of current cycle
                cycle  -> restart index (0,1,2,...)
            """
            num -= warm_up_num

            if T_mult == 1:
                cycle = num // T_0
                T_cur = num % T_0
                return T_cur, T_0, cycle

            T_i = T_0
            cycle = 0
            while num >= T_i:
                num -= T_i
                T_i *= T_mult
                cycle += 1

            return num, T_i, cycle

        self.eval_interval = lambda epoch: 1 if (epoch+1) > max_epoch - 10 else 5

        def lr_lambda(num):
            if num < warm_up_num:
                return num / warm_up_num

            T_cur, T_i, cycle = get_cycle_info(num)

            # decay the peak LR after each restart
            max_lr_factor = self.gamma ** cycle

            cosine = 0.5 * (1 + np.cos(np.pi * T_cur / T_i))

            return self.eta_min + (max_lr_factor - self.eta_min) * cosine

        self.lr_lambda = lr_lambda

    def get_lambda(self):
        return self.eval_interval, self.lr_lambda