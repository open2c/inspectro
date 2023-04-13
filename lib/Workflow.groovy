#!/usr/bin/env nextflow
nextflow.enable.dsl=2

class WorkflowMain {

    public static String getGenomeAttribute(params, attribute) {
        if (params.genomes && params.genome && params.genomes.containsKey(params.genome)) {
            if (params.genomes[ params.genome ].containsKey(attribute)) {
                return params.genomes[ params.genome ][ attribute ]
            }
        }
        return val
    }
}