package ok;

/**
* @author Petr Zhalybin
* @since 31.08.2014 13:35
*/
class WordStat {
    String word;
    //        int count;
    LikesStat likesStat = new LikesStat();
    int bowId = -1;
    double idf;

    public String toString() {
        return word;
    }
}
