package ok;

/**
* @author Petr Zhalybin
* @since 31.08.2014 13:34
*/
class LikesStat {
    int count;
    double likesum;
    double likesum2;
    double loglikesum;
    double loglikesum2;

    public void addPost(Post post) {
        count++;
        likesum += post.likes;
        likesum2 += post.likes * post.likes;
        loglikesum += post.loglikes;
        loglikesum2 += post.loglikes * post.loglikes;
    }
}
