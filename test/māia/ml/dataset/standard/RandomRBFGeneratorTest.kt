package māia.ml.dataset.standard

import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Tests the [RandomRBFGenerator] implementation.
 *
 * @author Corey Sterling (csterlin at waikato dot ac dot nz)
 */
class RandomRBFGeneratorTest {


    @Test
    fun testHeaders() {
        val gen = RandomRBFGenerator()

        assertEquals(11, gen.numColumns)

    }

}
