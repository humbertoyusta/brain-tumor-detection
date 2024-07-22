import prefect
import preprocessing.data_collector


@prefect.task
def collect_data():
    collector = preprocessing.data_collector.DataCollector()
    collector.run()
