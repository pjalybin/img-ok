package ok;

import java.io.Serializable;

/**
* @author Petr Zhalybin
* @since 31.08.2014 13:36
*/
interface Predictor extends Serializable {
    double predict(Post post);
}
