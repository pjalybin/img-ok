package ok;

import java.util.Collection;
import java.util.Map;

/**
 * @author Petr Zhalybin
 * @since  31.08.2014 13:32
 */
public interface Trainer {

   Predictor train(Collection<Post> posts, Parameters parameters, String trainId,
                                   Map<Integer, Post> testPosts, Predictor initialPred, int epochNum,
                                   final Map<Integer, Predictor> groupPred, int groupPredKey, Collection<Post> devCrossvalidation);
}
