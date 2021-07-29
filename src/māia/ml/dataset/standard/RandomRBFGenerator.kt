package māia.ml.dataset.standard

import māia.ml.dataset.DataColumnHeader
import māia.ml.dataset.DataMetadata
import māia.ml.dataset.DataRow
import māia.ml.dataset.DataStream
import māia.ml.dataset.type.DataType
import māia.ml.dataset.type.standard.NominalDoubleIndexImpl
import māia.ml.dataset.type.standard.NumericDoubleImpl
import māia.ml.dataset.util.buildRow
import māia.ml.dataset.util.formatStringSimple
import māia.util.magnitude
import māia.util.nextDoubleArray
import māia.util.nextGaussian
import māia.util.nextIntWeighted
import kotlin.random.Random

/**
 * TODO: What class does.
 *
 * @author Corey Sterling (csterlin at waikato dot ac dot nz)
 */
class RandomRBFGenerator(
    modelSeed: Int = 1,
    instanceSeed: Int = 1,
    private val numClasses: Int = 2,
    private val numAttributes: Int = 10,
    numCentroids: Int = 50
) : DataStream<DataRow> {

    private val centroids: Array<Centroid>

    private val centroidWeights: DoubleArray

    private val instanceRandom : Random = Random(instanceSeed)

    override val metadata : DataMetadata = object : DataMetadata {
        override val name : String
            get() = this@RandomRBFGenerator.toString()
    }

    override val numColumns : Int = numAttributes + 1

    init {
        val modelRand = Random(modelSeed)
        centroids = Array(numCentroids) {
            Centroid(
                modelRand.nextDoubleArray(numAttributes),
                modelRand.nextInt(numClasses).toDouble(),
                modelRand.nextDouble()
            )
        }
        centroidWeights = modelRand.nextDoubleArray(numCentroids)
    }

    private val headers: Array<DataColumnHeader> = Array(numColumns) { index ->
        if (index == numAttributes) {
            object : DataColumnHeader {
                override val name : String = "class"
                override val type : DataType<*, *> = NominalDoubleIndexImpl(*Array(numClasses) { "class ${it + 1}" })
                override val isTarget : Boolean = true
            }
        } else {
            object : DataColumnHeader {
                override val name : String = "att ${index + 1}"
                override val type : DataType<*, *> = NumericDoubleImpl
                override val isTarget : Boolean = false
            }
        }
    }

    override fun getColumnHeader(columnIndex : Int) : DataColumnHeader = headers[columnIndex]

    override fun rowIterator() : Iterator<DataRow> {
        return object : Iterator<DataRow> {
            override fun hasNext() : Boolean = true
            override fun next() : DataRow = this@RandomRBFGenerator.nextRow()
        }
    }

    private fun nextRow(): DataRow {
        val centroid = centroids[instanceRandom.nextIntWeighted(centroidWeights)]
        val attVals = DoubleArray(numColumns)
        for (index in 0 until numAttributes)
            attVals[index] = instanceRandom.nextDouble(-1.0, 1.0)
        val desiredMag = instanceRandom.nextGaussian() * centroid.stdDev
        val scale = desiredMag / attVals.magnitude
        for (index in 0 until numAttributes)
            attVals[index] = attVals[index] * scale + centroid.centre[index]

        attVals[numAttributes] = centroid.classLabel

        return buildRow(attVals.toTypedArray())
    }
}

private data class Centroid(
    val centre: DoubleArray,
    val classLabel: Double,
    val stdDev: Double
)
