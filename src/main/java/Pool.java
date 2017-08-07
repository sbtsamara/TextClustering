import ru.stachek66.nlp.mystem.holding.Factory;
import ru.stachek66.nlp.mystem.holding.MyStem;
import scala.Option;

import java.io.File;
import java.util.HashSet;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Created by azaz on 27.07.17.
 */
public class Pool {
    static private Pool instance = null;
    static private HashSet<MyStem> locked, unlocked;
    static ReentrantLock lock = new ReentrantLock();

    public static Pool getInstance(int count) {
        if (instance == null) {
            instance = new Pool(count);
        }
        return instance;
    }


    private Pool(int count) {
//        lock.unlock();
        this.locked = new HashSet<MyStem>();
        this.unlocked = new HashSet<MyStem>();
        for (int i = 0; i < count; i++) {
            unlocked.add(new Factory("-igd --eng-gr --format json --weight")
                    .newMyStem("3.0", Option.<File>empty()).get());
        }
    }

    public MyStem getObject() {
        try {
            lock.tryLock(5, TimeUnit.SECONDS);
            while (unlocked.size() == 0) {
                lock.unlock();
                Thread.sleep(300);
                lock.tryLock(5, TimeUnit.SECONDS);
            }
            MyStem item = unlocked.iterator().next();
            unlocked.remove(item);
            locked.add(item);
            lock.unlock();
            return item;


        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return null;
    }

    public void returnObject(MyStem o) {
//        System.out.println("return");
        try {
            lock.tryLock(5, TimeUnit.SECONDS);
            locked.remove(o);
            unlocked.add(o);
            lock.unlock();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

    }

}
