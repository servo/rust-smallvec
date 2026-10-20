use proc_macro::{
    Delimiter,
    TokenStream,
    TokenTree
};

#[proc_macro_attribute]
pub fn syntax(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut element = Vec::<TokenTree>::new();
    let mut iterator = item.into_iter().fuse();
    loop {
        match [iterator.next(), iterator.next()] {
            [Some(TokenTree::Punct(p)), Some(TokenTree::Group(g))]
                if (p.as_char(), g.delimiter()) == ('#', Delimiter::Bracket) =>
            {
                element.extend([p.into(), g.into()]);
            }
            mut trees => {
                let trees = trees.iter_mut().flat_map(Option::take);
                element.extend(attr.into_iter().chain(trees).chain(iter));
                return element.into_iter().collect();
            }
        };
    }
}
