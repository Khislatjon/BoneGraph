//
//  PaperListView.swift
//  PaperScraper
//
//  Created by Khislatjon Valijonov on 20/07/2025.
//

import SwiftUI

struct PaperListView: View {
    @State var viewModel = FetchPaperViewModel()
    
    var body: some View {
        List {
            ForEach(viewModel.papers, id: \.link) { paper in
                VStack(alignment: .leading, spacing: 8) {
                    Text("**Title:** \(paper.title)")
                    Text("**Authors:** \(paper.authors.joined(separator: ", "))")
                    Text("**Year:** \(String(paper.year))")
                }.multilineTextAlignment(.leading)
            }
        }
        .padding()
        .onAppear {
            Task {
                await viewModel.downloadArxivPDFs(for: "machine learning")
            }
        }
    }
}

#Preview {
    PaperListView()
}
