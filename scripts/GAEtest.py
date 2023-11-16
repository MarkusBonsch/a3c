import mxnet as mx

class GAEtest:
    def __init__(self, values = [1,3,2,4], rewards = [6,4,5,3], resetTrigger = [0,0,0,0], ga = 0.99):
        self.values = mx.nd.array(values)
        self.rewards = mx.nd.array(rewards)
        self.resetTrigger = resetTrigger
        self.rewardDiscount = ga

    def getDiscountedRewardNormal(self, lastValue):
        """
        calculates the reward
        Args:
            lastValue(float): value of step after last action
        Returns:
            (mx.nd.array) discounted reward for each timestep 
        """
        timeLeft = self.rewards.shape[0]
        R = lastValue
        out = mx.nd.zeros(shape = (1))
        for n in reversed(range(timeLeft)):
            if self.resetTrigger[n]: R = 0
            R = self.rewardDiscount * R + self.rewards[n]
            out = mx.nd.concat(out, R, dim = 0)
        ## delete dummy first element and reverse again
        out = out[1:]
        out = out[::-1]
        return(out)

    def getDiscountedReward(self, lastValue, la = 0.96):
        """
        calculates the advantage according to generalized advantage estimation
        Args:
           lastValue(float): value of step after last action
           la: hyperparameter. Make it 1 for normal reward estimation
        Returns:
           discounted reward
        """
        ### calculates advantage first using GAE. Then in the last step reward is calculated
        timeLeft = self.rewards.shape[0]
        GAE    = 0
        ndLastValue = mx.nd.zeros(shape = (1))
        ndLastValue[0] = lastValue
        values = mx.nd.concat(self.values, ndLastValue, dim = 0)
        out    = mx.nd.zeros(shape = (1))
        for n in reversed(range(timeLeft)):
            if self.resetTrigger[n]: 
                GAE = 0
                values[n+1] = 0 ## value after game ends is 0
            delta = self.rewards[n] + self.rewardDiscount * values[n+1] - values[n]
            GAE   = self.rewardDiscount * la * GAE + delta
            out   = mx.nd.concat(out, GAE, dim = 0)
        ## delete dummy first element and reverse again
        out = out[1:]
        out = out[::-1]
        ## add values to obtain rewards instead of advantages
        out = out + self.values
        return(out)