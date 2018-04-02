import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.junit.Test;

import java.util.Arrays;
import java.util.LinkedList;

/**
 * Created with IntelliJ IDEA.
 * User: sssd
 * Date: 2018/4/2 16:22
 * Version: V1.0
 * To change this template use File | Settings | File Templates.
 * Description:
 */
public class TestUtils {

    @Test
    public void testList(){
        LinkedList<Double> list = new LinkedList<Double>();
        list.add(1.0);
        list.add(2.0);
        list.add(3.0);
        Double[] doubles1 = new Double[4];
        list.toArray(doubles1);
        System.out.println(Arrays.asList(doubles1));
    }

    @Test
    public void testParse(){
        String str = "1.0,2.0,3.0";
        Vector vector = Vectors.parse(str);
        System.out.println(vector.toJson());
    }
}
